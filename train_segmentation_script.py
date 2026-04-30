# Wound Segmentation Training
# EfficientNet-B3 + Full Pipeline + Early Stopping + Plotting

import os, sys, time, argparse, traceback, random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# These limits keep CPU-side libraries from starting too many helper threads.
# On Windows, that can otherwise make training look "busy" without actually helping much.
# Keeping thread counts small usually makes runs more stable and logs easier to read.
cv2.setNumThreads(0)                # prevent OpenCV thread spam
os.environ["OMP_NUM_THREADS"] = "1" # limit OpenMP threads
torch.set_num_threads(1)

# make stdout flush immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# arguments
def get_args():
    p = argparse.ArgumentParser(description="Wound U-Net training")
    p.add_argument("--ram-cache", action="store_true",
                   help="preload images/masks in RAM for faster epochs")
    p.add_argument("--accum", type=int, default=1,
                   help="gradient accumulation steps (1 = off)")
    p.add_argument("--bf16", action="store_true",
                   help="use bfloat16 autocast instead of float16")
    p.add_argument("--data-root",   required=True,
                   help="path to folder of original images")
    p.add_argument("--mask-root",   required=True,
                   help="path to folder of mask images")
    p.add_argument("--save-path",   required=True,
                   help="path/filename to save best model (e.g. ./best_model.pth)")
    p.add_argument("--img-size",    type=int, default=352)
    p.add_argument("--epochs",      type=int, default=500)
    p.add_argument("--patience",    type=int, default=20,
                   help="early stopping patience")
    p.add_argument("--batch",       type=int, default=8)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 is safest for Windows/Jupyter")
    p.add_argument("--seed",        type=int, default=1,
                   help="random seed")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay")
    return p.parse_args()


# Dataset (with Albumentations)
class WoundDataset(Dataset):
    def __init__(self, all_img_paths, all_mask_paths, img_size=352, ram_cache=False, is_val=False):
        self.img_size = img_size
        self.is_val = is_val
        self.ram_cache = ram_cache
        self.img_paths = all_img_paths
        self.mask_paths = all_mask_paths
        # Assertion expanded
        if len(self.img_paths) != len(self.mask_paths):
             raise AssertionError("Image and mask counts mismatch")

        self.imgs = [None] * len(self.img_paths)
        self.masks = [None] * len(self.mask_paths)

        if self.ram_cache:
            # If RAM cache is enabled, we load everything once up front and keep it in memory.
            # This makes each epoch faster because the dataset no longer has to read from disk
            # every time __getitem__ is called. The tradeoff is higher memory usage.
            print(f"RAM Caching {'Val' if is_val else 'Train'}:", end="")
            st = time.time()
            for i in range(len(self.img_paths)):
                 # Expanded print condition
                 if i % 50 == 0:
                     print(".", end="")
                 self.imgs[i] = self._load_img(i)
                 self.masks[i] = self._load_mask(i)
            print(f" done in {time.time() - st:.2f}s")

        if is_val:
            # Validation should be boring and consistent:
            # resize the sample, normalize it, convert to tensor.
            # We do NOT add random flips/rotations here, because validation is meant to
            # measure how good the current model really is, not how it behaves under random noise.
            self.transform = A.Compose([
                A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            # Training is where we intentionally add variation.
            # Flips, rotations, elastic warps, and color changes help the model learn
            # the wound concept instead of memorizing exact training images.
            # Albumentations applies the geometric transforms to both image and mask together,
            # which is important because the mask must stay perfectly aligned with the image.
            self.transform = A.Compose([
                A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.ElasticTransform(p=0.2, alpha=1, sigma=50, border_mode=cv2.BORDER_CONSTANT),
                A.GridDistortion(p=0.2, border_mode=cv2.BORDER_CONSTANT),
                A.ShiftScaleRotate(p=0.2, shift_limit=0.1, scale_limit=0.1, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
                A.ColorJitter(p=0.3, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def _load_img(self, idx):
        img = cv2.imread(str(self.img_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, idx):
        mask = cv2.imread(str(self.mask_paths[idx]), 0)
        return mask

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.imgs[idx] if self.ram_cache else self._load_img(idx)
        mask = self.masks[idx] if self.ram_cache else self._load_mask(idx)

        augmented = self.transform(image=img, mask=mask)
        img_tensor = augmented['image']
        mask_tensor = augmented['mask']

        # The mask image comes in as grayscale values like 0 and 255.
        # For training, we want a single-channel tensor with values in [0, 1],
        # so we divide by 255 and add a channel dimension with unsqueeze(0).
        mask_tensor = (mask_tensor.float() / 255.0).unsqueeze(0)

        return img_tensor, mask_tensor


# Loss / Metrics
class SafeDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SafeDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # The model outputs logits, which are raw scores.
        # We pass them through sigmoid first so they become probabilities between 0 and 1.
        # Dice compares overlap, so probabilities are easier to reason about than raw scores.
        logits = torch.sigmoid(logits)
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (logits_flat * targets_flat).sum()
        union = logits_flat.sum() + targets_flat.sum()
        
        # If both prediction and target are completely empty, the denominator becomes zero.
        # Returning 0 loss here means "there is no segmentation error to punish".
        if union == 0:
            return torch.tensor(0.0, device=logits.device)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class SafeFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(SafeFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Focal loss starts from binary cross-entropy, then gives extra attention
        # to harder examples. This is useful in segmentation because background pixels
        # often dominate, and easy negatives can otherwise overwhelm the loss.
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # Clamp keeps pt away from exact 0 or 1.
        # That avoids unstable math and helps keep gradients well-behaved.
        pt = torch.clamp(pt, 1e-6, 1-1e-6)
        
        focal_weight = (1 - pt).pow(self.gamma)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def safe_dice_coeff(logits, targets, smooth=1e-6):
    logits = torch.sigmoid(logits)
    logits_flat = logits.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (logits_flat * targets_flat).sum()
    union = logits_flat.sum() + targets_flat.sum()
    
    # Some images may contain no wound at all.
    # In that case, if the model also predicts "empty", we treat it as a perfect Dice score
    # instead of allowing a divide-by-zero situation.
    if union == 0:
        return torch.tensor(1.0, device=logits.device)  # Perfect score for empty masks
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# MAIN
def main():
    args = get_args()

    # setup
    EPOCHS = args.epochs
    BATCH = args.batch
    IMG_SIZE = args.img_size
    LR = args.lr
    SEED = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Seeding makes runs more repeatable.
    # That is especially helpful when debugging training, because you want fewer
    # "it changed just because randomness changed" moments.
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Device: {device} | Img Size: {IMG_SIZE} | Batch: {BATCH} | LR: {LR} | Seed: {SEED}")
    print(f"RAM Cache: {args.ram_cache} | Accum: {args.accum} | BF16: {args.bf16}")
    print(f"Weight Decay: {args.weight_decay}")

    print(f"Scanning for images in: {args.data_root} (recursive)")
    img_files = list(Path(args.data_root).glob("**/*.png"))
    img_files.extend(Path(args.data_root).glob("**/*.jpg"))
    img_files.extend(Path(args.data_root).glob("**/*.jpeg"))
    all_img_paths = sorted(img_files)

    print(f"Scanning for masks in: {args.mask_root} (recursive)")
    mask_files = list(Path(args.mask_root).glob("**/*.png"))
    all_mask_paths = sorted(mask_files)

    # Expanded checks
    if not all_img_paths:
        print(f"\n--- FATAL ERROR ---")
        print(f"No images (.png, .jpg, .jpeg) found in: {args.data_root}")
        return

    if not all_mask_paths:
        print(f"\n--- FATAL ERROR ---")
        print(f"No masks (.png) found in: {args.mask_root}")
        return

    print(f"Found {len(all_img_paths)} images and {len(all_mask_paths)} masks. Matching...")

    # We match files by their stem (filename without extension).
    # Example:
    #   photo_001.jpg   -> stem = "photo_001"
    #   photo_001_mask.png -> stem = "photo_001_mask"
    # This lets us pair masks to images even if they are stored in separate folders.
    img_map = {p.stem: p for p in all_img_paths}

    matched_img_paths = []
    matched_mask_paths = []

    for mask_p in all_mask_paths:
        mask_stem = mask_p.stem

        # Support two common naming styles:
        # 1) The mask adds a "_mask" suffix.
        # 2) The image and mask share the same stem.
        # This keeps the script flexible across slightly different dataset layouts.
        if mask_stem.endswith("_mask"):
            expected_img_stem = mask_stem[:-5]
            if expected_img_stem in img_map:
                matched_mask_paths.append(mask_p)
                matched_img_paths.append(img_map[expected_img_stem])
        elif mask_stem in img_map:
            matched_mask_paths.append(mask_p)
            matched_img_paths.append(img_map[mask_stem])

    # Expanded check
    if not matched_img_paths:
        print(f"\n--- FATAL ERROR ---")
        print(f"Found {len(all_img_paths)} images and {len(all_mask_paths)} masks, but 0 had matching filenames.")
        print(f"Example img stem: {all_img_paths[0].stem}")
        print(f"Example mask stem: {all_mask_paths[0].stem}")
        return

    print(f"Matched {len(matched_img_paths)} image/mask pairs")

    # Split only after matching pairs.
    # If we split before matching, we could accidentally create broken samples
    # where one side of the pair is missing.
    all_indices = list(range(len(matched_img_paths)))
    random.shuffle(all_indices)
    split = int(len(all_indices) * 0.8)
    train_indices = all_indices[:split]
    val_indices = all_indices[split:]

    print(f"Dataset: {len(all_indices)} matched image/mask pairs | Train: {len(train_indices)} | Val: {len(val_indices)}")

    # Create datasets with the matched pairs only
    train_ds = WoundDataset(
        [matched_img_paths[i] for i in train_indices],
        [matched_mask_paths[i] for i in train_indices], 
        IMG_SIZE, args.ram_cache, is_val=False
    )

    val_ds = WoundDataset(
        [matched_img_paths[i] for i in val_indices],
        [matched_mask_paths[i] for i in val_indices],
        IMG_SIZE, args.ram_cache, is_val=True
    )

    print(f"Final datasets - Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Validation can usually run with a bigger batch size because we are not doing backward()
    # there, so memory pressure is lower than during training.
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=BATCH*2, shuffle=False, num_workers=args.num_workers)

    print("[model] Using smp.Unet with efficientnet-b3 backbone")
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    opt = Adam(model.parameters(), lr=LR, weight_decay=args.weight_decay)

    T_MAX_EPOCHS = 50
    print(f"[scheduler] Using CosineAnnealingLR with T_max = {T_MAX_EPOCHS} epochs")
    sched = CosineAnnealingLR(opt, T_max=T_MAX_EPOCHS, eta_min=1e-6)

    print("[loss] Using 0.5 * SafeFocalLoss + 0.5 * SafeDiceLoss")
    focal = SafeFocalLoss()
    dloss = SafeDiceLoss()
    # GradScaler protects fp16 training from underflow by scaling the loss before backward().
    # For bf16 this is usually unnecessary, so we disable scaling when bf16 is selected.
    scaler = torch.cuda.amp.GradScaler(enabled=not args.bf16)
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    best_dice = 0.0
    bad = 0

    train_loss_hist = []
    val_loss_hist = []
    val_dice_hist = []

    # TRAINING LOOP
    print(f"Starting training for {EPOCHS} epochs...")
    for ep in range(1, EPOCHS + 1):
        model.train()
        tl, td = 0.0, 0.0
        st = time.time()

        if ep == 1:
            print("[warmup] loading first batch...")
            first_batch_st = time.time()

        for i, (x, y) in enumerate(train_dl):
            if ep == 1 and i == 0:
                print(f"[warmup] first batch loaded in {time.time() - first_batch_st:.2f}s")
                st = time.time()

            x, y = x.to(device), y.to(device)

            # autocast tells PyTorch it is allowed to use lower-precision math where safe.
            # This often reduces memory use and speeds training on modern GPUs.
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(x)
                loss = 0.5 * focal(logits, y) + 0.5 * dloss(logits, y)
                # If gradient accumulation is enabled, divide the loss here.
                # That way, accumulating N smaller mini-batches behaves like training
                # on one larger effective batch.
                loss /= args.accum

            scaler.scale(loss).backward()

            # We do not update weights after every mini-batch when accumulation is active.
            # Instead, we wait until enough mini-batches have contributed gradients,
            # then perform a single optimizer step.
            if ((i + 1) % args.accum == 0) or (i + 1 == len(train_dl)):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            tl += (loss.item() * args.accum) * x.size(0)
            td += safe_dice_coeff(logits, y).item() * x.size(0)

            # Expanded print condition
            if i > 0 and (i % (max(1, len(train_dl) // 10)) == 0 or i == len(train_dl) - 1):
                print(f"  batch {i+1}/{len(train_dl)} done in {time.time()-st:.2f}s")
                st = time.time()

        tl /= len(train_ds)
        td /= len(train_ds)

        # Validation is inference-only:
        # no gradient tracking, no optimizer updates, just measure loss and Dice.
        model.eval()
        vl, vd = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(x)
                    vloss  = 0.5 * focal(logits, y) + 0.5 * dloss(logits, y)
                vl += vloss.item() * x.size(0)
                vd += safe_dice_coeff(logits, y).item() * x.size(0)

        vl /= len(val_ds)
        vd /= len(val_ds)

        # Append metrics to history lists
        train_loss_hist.append(tl)
        val_loss_hist.append(vl)
        val_dice_hist.append(vd)

        cur_lr = opt.param_groups[0]["lr"]
        print(f"Ep {ep:03d}/{EPOCHS} | Train {tl:.4f} D{td:.4f} | Val {vl:.4f} D{vd:.4f} | lr={cur_lr:.6g}", flush=True)

        sched.step()

        # Early stopping watches validation Dice instead of training loss.
        # That is a better signal for segmentation quality, because a lower training loss
        # does not always mean the model is generalizing better.
        if vd > best_dice:
            best_dice = vd
            bad = 0
            torch.save(model.state_dict(), save_path)
            print(f"[seg] New best Dice = {best_dice:.4f} - saved {save_path}", flush=True)
        else:
            bad += 1
            print(f"[seg] no improvement for {bad}/{args.patience} epochs", flush=True)
            if bad >= args.patience:
                print("[seg] Early stopping.", flush=True)
                break

    print("Training complete.", flush=True)

    # --- Plotting ---
    epochs_ran = len(train_loss_hist)
    epochs_axis = range(1, epochs_ran + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss (Expanded)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_axis, train_loss_hist, label="Train Loss")
    plt.plot(epochs_axis, val_loss_hist, label="Val Loss")
    plt.title("Segmentation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot Dice (Expanded)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_axis, val_dice_hist, label="Val Dice", color='green')
    plt.title("Validation Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    # Highlight best Dice score
    if val_dice_hist: # Check if list is not empty
        best_epoch_idx = np.argmax(val_dice_hist)
        plt.scatter(best_epoch_idx + 1, best_dice, color='red', zorder=5, label=f"Best: {best_dice:.4f}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plot_save_path = "segmentation_curves.png"
    plt.savefig(plot_save_path, dpi=150)
    print(f"Saved training curves to {plot_save_path}")


# BOOTSTRAP (Windows workers need spawn under __main__)
if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp
        # Windows does not support the Linux-style "fork" behavior used in many PyTorch examples.
        # Setting "spawn" here avoids worker startup issues when DataLoader uses subprocesses.
        mp.set_start_method('spawn')
    except RuntimeError:
        pass # Already set or not available
    except Exception as e: # Catch other potential exceptions
        print(f"[warn] Failed to set multiprocessing start method: {e}")

    try:
        main()
    except Exception as e:
        print("\n--- ERROR ---")
        traceback.print_exc()
        print("-------------")
