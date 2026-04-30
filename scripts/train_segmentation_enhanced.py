"""
Wound Segmentation Training Script (Enhanced for FYP Report)
Run: python -u train_segmentation_enhanced.py --data-root ... --mask-root ... --save-dir ... --exp-name S1_baseline

Features:
  - CSV logging of all metrics per epoch
  - Training curves (loss + Dice)
  - Sample prediction visualizations
  - IoU and Dice metrics
  - Experiment naming for organized outputs

Key flags:
  --exp-name       : experiment name (e.g., S1_baseline, S2_highres)
  --ram-cache      : preload images into RAM for faster epochs
  --accum N        : gradient accumulation steps
  --bf16           : use bfloat16 autocast (RTX 40-series)
"""

import os, sys, time, argparse, traceback, random, json, csv
from pathlib import Path
from datetime import datetime
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

cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def get_args():
    p = argparse.ArgumentParser(description="Wound U-Net training (Enhanced)")
    
    # Required
    p.add_argument("--data-root", required=True, help="Path to folder of original images")
    p.add_argument("--mask-root", required=True, help="Path to folder of mask images")
    p.add_argument("--save-path", required=True, help="Path to save best model (e.g., D:/WoundCare/experiments/S1_baseline/best_model.pth)")
    
    # Training
    p.add_argument("--img-size", type=int, default=352)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    
    # Performance
    p.add_argument("--ram-cache", action="store_true", help="Preload images/masks in RAM")
    p.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast")
    
    return p.parse_args()


class WoundDataset(Dataset):
    """Dataset for wound segmentation with optional RAM caching"""
    def __init__(self, img_paths, mask_paths, img_size=352, ram_cache=False, is_val=False):
        self.img_size = img_size
        self.is_val = is_val
        self.ram_cache = ram_cache
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        
        if len(self.img_paths) != len(self.mask_paths):
            raise AssertionError("Image and mask counts mismatch")
        
        self.imgs = [None] * len(self.img_paths)
        self.masks = [None] * len(self.mask_paths)
        
        if self.ram_cache:
            print(f"RAM Caching {'Val' if is_val else 'Train'}:", end="", flush=True)
            st = time.time()
            for i in range(len(self.img_paths)):
                if i % 50 == 0:
                    print(".", end="", flush=True)
                self.imgs[i] = self._load_img(i)
                self.masks[i] = self._load_mask(i)
            print(f" done in {time.time() - st:.2f}s")
        
        if is_val:
            self.transform = A.Compose([
                A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
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
        mask_tensor = (mask_tensor.float() / 255.0).unsqueeze(0)
        
        return img_tensor, mask_tensor


class SafeDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (logits_flat * targets_flat).sum()
        union = logits_flat.sum() + targets_flat.sum()
        
        if union == 0:
            return torch.tensor(0.0, device=logits.device)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class SafeFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        pt = torch.clamp(pt, 1e-6, 1-1e-6)
        
        focal_weight = (1 - pt).pow(self.gamma)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def compute_metrics(logits, targets, threshold=0.5):
    """Compute Dice and IoU metrics"""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    union_dice = preds_flat.sum() + targets_flat.sum()
    union_iou = preds_flat.sum() + targets_flat.sum() - intersection
    
    # Dice
    if union_dice == 0:
        dice = 1.0  # Both empty
    else:
        dice = (2. * intersection + 1e-6) / (union_dice + 1e-6)
    
    # IoU
    if union_iou == 0:
        iou = 1.0
    else:
        iou = (intersection + 1e-6) / (union_iou + 1e-6)
    
    return float(dice), float(iou)


def plot_training_curves(train_loss, val_loss, val_dice, val_iou, save_path):
    """Plot and save training curves"""
    epochs = range(1, len(train_loss) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Dice
    axes[1].plot(epochs, val_dice, 'g-', label='Val Dice', linewidth=2)
    best_epoch = np.argmax(val_dice) + 1
    best_dice = max(val_dice)
    axes[1].scatter([best_epoch], [best_dice], color='red', s=100, zorder=5, 
                    label=f'Best: {best_dice:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Validation Dice Score')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # IoU
    axes[2].plot(epochs, val_iou, 'm-', label='Val IoU', linewidth=2)
    best_iou = max(val_iou)
    axes[2].scatter([np.argmax(val_iou) + 1], [best_iou], color='red', s=100, zorder=5,
                    label=f'Best: {best_iou:.4f}')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU Score')
    axes[2].set_title('Validation IoU Score')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_sample_predictions(model, val_ds, device, save_dir, num_samples=6):
    """Save sample prediction visualizations"""
    model.eval()
    indices = np.random.choice(len(val_ds), min(num_samples, len(val_ds)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, mask_tensor = val_ds[idx]
            
            # Get prediction
            img_input = img_tensor.unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type=="cuda"):
                logits = model(img_input)
            pred = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Denormalize image
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # Mask
            mask_np = mask_tensor.squeeze().cpu().numpy()
            
            # Binary prediction
            pred_binary = (pred > 0.5).astype(np.float32)
            
            # Compute metrics for this sample
            dice, iou = compute_metrics(
                torch.from_numpy(pred).unsqueeze(0).unsqueeze(0),
                mask_tensor.unsqueeze(0)
            )
            
            # Plot
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='jet', vmin=0, vmax=1)
            axes[i, 2].set_title('Prediction (Probability)')
            axes[i, 2].axis('off')
            
            # Overlay
            overlay = img_np.copy()
            mask_overlay = np.zeros_like(overlay)
            mask_overlay[:, :, 1] = (pred_binary * 255).astype(np.uint8)  # Green for prediction
            mask_overlay[:, :, 0] = (mask_np * 255).astype(np.uint8)  # Red for ground truth
            overlay = cv2.addWeighted(overlay, 0.6, mask_overlay, 0.4, 0)
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f'Overlay (Dice={dice:.3f}, IoU={iou:.3f})')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / "sample_predictions.png", dpi=150)
    plt.close()


def main():
    args = get_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determinism
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create output directory (use parent of save_path)
    save_path = Path(args.save_path)
    exp_dir = save_path.parent
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_name = exp_dir.name  # Use folder name as experiment name
    
    print("="*70)
    print(f"WOUND SEGMENTATION TRAINING - {exp_name}")
    print(f"Output directory: {exp_dir}")
    print(f"Image size: {args.img_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {device}")
    print(f"RAM Cache: {args.ram_cache} | Accum: {args.accum} | BF16: {args.bf16}")
    print("="*70, flush=True)
    
    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config['exp_name'] = exp_name
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Find images and masks
    print(f"\nScanning for images in: {args.data_root}")
    img_files = list(Path(args.data_root).glob("**/*.png"))
    img_files += list(Path(args.data_root).glob("**/*.jpg"))
    img_files += list(Path(args.data_root).glob("**/*.jpeg"))
    all_img_paths = sorted(img_files)
    
    print(f"Scanning for masks in: {args.mask_root}")
    mask_files = list(Path(args.mask_root).glob("**/*.png"))
    all_mask_paths = sorted(mask_files)
    
    if not all_img_paths:
        print(f"FATAL: No images found in {args.data_root}")
        return
    if not all_mask_paths:
        print(f"FATAL: No masks found in {args.mask_root}")
        return
    
    print(f"Found {len(all_img_paths)} images and {len(all_mask_paths)} masks. Matching...")
    
    # Match images and masks
    img_map = {p.stem: p for p in all_img_paths}
    matched_img_paths = []
    matched_mask_paths = []
    
    for mask_p in all_mask_paths:
        mask_stem = mask_p.stem
        if mask_stem.endswith("_mask"):
            expected_img_stem = mask_stem[:-5]
            if expected_img_stem in img_map:
                matched_mask_paths.append(mask_p)
                matched_img_paths.append(img_map[expected_img_stem])
        elif mask_stem in img_map:
            matched_mask_paths.append(mask_p)
            matched_img_paths.append(img_map[mask_stem])
    
    if not matched_img_paths:
        print(f"FATAL: No matching image/mask pairs found")
        return
    
    print(f"Matched {len(matched_img_paths)} image/mask pairs")
    
    # Split dataset
    all_indices = list(range(len(matched_img_paths)))
    random.shuffle(all_indices)
    split = int(len(all_indices) * 0.8)
    train_indices = all_indices[:split]
    val_indices = all_indices[split:]
    
    print(f"Dataset split: Train={len(train_indices)} | Val={len(val_indices)}")
    
    # Create datasets
    train_ds = WoundDataset(
        [matched_img_paths[i] for i in train_indices],
        [matched_mask_paths[i] for i in train_indices],
        args.img_size, args.ram_cache, is_val=False
    )
    val_ds = WoundDataset(
        [matched_img_paths[i] for i in val_indices],
        [matched_mask_paths[i] for i in val_indices],
        args.img_size, args.ram_cache, is_val=True
    )
    
    # DataLoaders
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch*2, shuffle=False, num_workers=args.num_workers)
    
    # Model
    print("\n[Model] U-Net with EfficientNet-B3 backbone")
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    
    # Optimizer and scheduler
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    T_MAX = 50
    print(f"[Scheduler] CosineAnnealingLR with T_max={T_MAX}")
    sched = CosineAnnealingLR(opt, T_max=T_MAX, eta_min=1e-6)
    
    # Loss
    print("[Loss] 0.5 * FocalLoss + 0.5 * DiceLoss")
    focal = SafeFocalLoss()
    dloss = SafeDiceLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=not args.bf16 and device.type == "cuda")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    
    # CSV logging
    csv_path = exp_dir / "training_log.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_dice', 'val_loss', 'val_dice', 'val_iou', 'learning_rate', 'best'])
    
    # Training history
    train_loss_hist, val_loss_hist = [], []
    train_dice_hist, val_dice_hist, val_iou_hist = [], [], []
    
    best_dice = 0.0
    bad = 0
    start_time = time.time()
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*70, flush=True)
    
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, td = 0.0, 0.0
        
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type=="cuda"):
                logits = model(x)
                loss = 0.5 * focal(logits, y) + 0.5 * dloss(logits, y)
                loss /= args.accum
            
            scaler.scale(loss).backward()
            
            if ((i + 1) % args.accum == 0) or (i + 1 == len(train_dl)):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            
            tl += (loss.item() * args.accum) * x.size(0)
            dice, _ = compute_metrics(logits.detach(), y)
            td += dice * x.size(0)
        
        tl /= len(train_ds)
        td /= len(train_ds)
        
        # Validation
        model.eval()
        vl, vd, vi = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type=="cuda"):
                    logits = model(x)
                    vloss = 0.5 * focal(logits, y) + 0.5 * dloss(logits, y)
                vl += vloss.item() * x.size(0)
                dice, iou = compute_metrics(logits, y)
                vd += dice * x.size(0)
                vi += iou * x.size(0)
        
        vl /= len(val_ds)
        vd /= len(val_ds)
        vi /= len(val_ds)
        
        # Record history
        train_loss_hist.append(tl)
        train_dice_hist.append(td)
        val_loss_hist.append(vl)
        val_dice_hist.append(vd)
        val_iou_hist.append(vi)
        
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        
        is_best = vd > best_dice
        print(f"Ep {ep:03d}/{args.epochs} | Train {tl:.4f} D{td:.4f} | Val {vl:.4f} D{vd:.4f} IoU{vi:.4f} | lr={cur_lr:.6g}", end="")
        
        # Log to CSV
        csv_writer.writerow([ep, f"{tl:.6f}", f"{td:.6f}", f"{vl:.6f}", f"{vd:.6f}", f"{vi:.6f}", f"{cur_lr:.8f}", "1" if is_best else "0"])
        csv_file.flush()
        
        if is_best:
            best_dice = vd
            bad = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': ep,
                'best_dice': best_dice,
                'config': config
            }, save_path)
            print(f"   Best!", flush=True)
        else:
            bad += 1
            print(f"  | No improve: {bad}/{args.patience}", flush=True)
            if bad >= args.patience:
                print("\n>>> Early stopping.", flush=True)
                break
    
    csv_file.close()
    elapsed = time.time() - start_time
    
    print("="*70)
    print(f"Training completed in {elapsed/60:.1f} min")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print("="*70)
    
    # Generate visualizations
    print("\nGenerating reports and visualizations...")
    
    # 1. Training curves
    plot_training_curves(train_loss_hist, val_loss_hist, val_dice_hist, val_iou_hist,
                         exp_dir / "training_curves.png")
    print(f"  Saved: training_curves.png")
    
    # 2. Sample predictions (load best model first)
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    save_sample_predictions(model, val_ds, device, exp_dir, num_samples=6)
    print(f"  Saved: sample_predictions.png")
    
    # 3. Summary
    summary = {
        "experiment": exp_name,
        "model": "U-Net + EfficientNet-B3",
        "best_dice": float(best_dice),
        "best_iou": float(max(val_iou_hist)),
        "epochs_trained": len(train_loss_hist),
        "training_time_min": elapsed / 60,
        "final_train_loss": float(train_loss_hist[-1]),
        "final_val_loss": float(val_loss_hist[-1]),
        "dataset_size": len(matched_img_paths),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "config": config
    }
    with open(exp_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: summary.json")
    
    print(f"\n All outputs saved to: {exp_dir}")
    print("="*70)


if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    except Exception as e:
        print(f"[warn] Failed to set multiprocessing start method: {e}")
    
    try:
        main()
    except Exception as e:
        print("\n--- ERROR ---")
        traceback.print_exc()
        print("-------------")
