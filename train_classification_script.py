"""
Wound Classification Training Script (Windows-friendly, fast)
Run: python -u train_classification.py --crop-root ... --save-path ...

Key flags:
  --ram-cache      : preload images into RAM for faster epochs (safe with workers=0)
  --accum N        : gradient accumulation steps (simulate larger batch)
  --bf16           : use bfloat16 autocast (great on RTX 40-series)
  --num-workers 0  : safe on Windows/Jupyter
"""

import os, sys, time, math, argparse, random
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2 # Needed for albumentations
import pandas as pd
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import datasets, models # Keep torchvision.datasets for ImageFolder
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# system/thread knobs
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# args
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crop-root",   required=False)
    p.add_argument("--save-path",   required=True)
    p.add_argument("--img-size",    type=int, default=256)
    p.add_argument("--batch",       type=int, default=32)
    p.add_argument("--epochs",      type=int, default=120)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience",    type=int, default=15)
    p.add_argument("--label-smooth",type=float, default=0)
    p.add_argument("--weight-decay",type=float, default=1e-4)
    p.add_argument("--warmup-epochs",type=int, default=5)
    p.add_argument("--mixup-alpha", type=float, default=0)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--ram-cache",   action="store_true", help="preload images into RAM")
    p.add_argument("--accum",       type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--bf16",        action="store_true", help="use bfloat16 autocast")
    p.add_argument("--val-ratio",   type=float, default=0.3, help="val split ratio for non-kfold")
    p.add_argument("--kfolds",      type=int, default=0, help="number of folds for stratified kfold")
    p.add_argument("--fold-index",  type=int, default=0, help="1-based fold index for kfold")
    p.add_argument("--train-csv",   default="", help="CSV path for train split (uses CSV dataset)")
    p.add_argument("--val-csv",     default="", help="CSV path for val split (uses CSV dataset)")
    p.add_argument("--splits-root", default="", help="Root folder with fold_*/roi_gt_train.csv, roi_gt_val.csv")
    p.add_argument("--image-col",   default="image", help="CSV column for image filename/path")
    p.add_argument("--label-col",   default="stage", help="CSV column for label")
    p.add_argument("--images-root", default="", help="Base directory for CSV image paths")
    p.add_argument("--exclude-dir", default="", help="Folder of excluded image filenames (safety check)")
    return p.parse_args()

# tiny, sklearn-free split
def simple_split(n, val_ratio=0.3, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1 - val_ratio))
    return idx[:cut].tolist(), idx[cut:].tolist()

def stratified_kfold_indices(targets, n_splits, seed=42):
    rng = np.random.default_rng(seed)
    class_to_idx = {}
    for i, y in enumerate(targets):
        class_to_idx.setdefault(int(y), []).append(i)
    for idxs in class_to_idx.values():
        rng.shuffle(idxs)
    folds = [[] for _ in range(n_splits)]
    for idxs in class_to_idx.values():
        for i, idx in enumerate(idxs):
            folds[i % n_splits].append(idx)
    for f in folds:
        rng.shuffle(f)
    return folds

def print_dist(label, targets, classes):
    counts = np.bincount(targets, minlength=len(classes))
    print(f"{label} distribution:")
    for i, c in enumerate(classes):
        print(f"  {c}: {counts[i]}")

def load_exclude_set(exclude_dir):
    if not exclude_dir:
        return set()
    exts = {".jpg", ".jpeg", ".png"}
    root = Path(exclude_dir)
    if not root.exists():
        print(f"Warning: exclude dir not found: {exclude_dir}")
        return set()
    return {p.name.lower() for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts}

def assert_no_excluded(samples, exclude_set, label):
    if not exclude_set:
        return
    hits = [Path(p).name.lower() for p, _ in samples if Path(p).name.lower() in exclude_set]
    if hits:
        uniq = sorted(set(hits))
        example = ", ".join(uniq[:5])
        raise ValueError(f"{label} contains {len(hits)} excluded images. Examples: {example}")
    print(f"{label}: 0 excluded images (checked {len(exclude_set)} filenames)")

# RAM-cached dataset
class RAMImageDataset(Dataset):
    def __init__(self, samples, classes, img_size, transform):
        self.samples = samples
        self.classes = classes
        self.H = self.W = img_size
        self.transform = transform
        self.data = []
        self.targets = []
        print(f"RAM Caching {len(samples)} images:", end="")
        st = time.time()
        for i, (p, y) in enumerate(samples):
            if i % 100 == 0:
                print(".", end="")
            img = cv2.imread(str(p))
            if img is None:
                print(f"\nWarning: Skipping unloadable image {p}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            self.data.append(img)
            self.targets.append(y)
        self.targets = np.array(self.targets, dtype=np.int64)
        print(f" done in {time.time() - st:.2f}s")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img_np = self.data[i]
        target = int(self.targets[i])
        if self.transform:
            augmented = self.transform(image=img_np)
            img_tensor = augmented['image']
        else:
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        return img_tensor, target

class CSVImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=img)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img_tensor, int(y)

# mixup
def mixup(x, y, alpha):
    if alpha <= 0:
        return x, None, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1 - lam) * x[idx]
    return x, (y, y[idx], lam), idx

# main
def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("="*70)
    print("WOUND CLASSIFICATION TRAINING (EfficientNet-B5 + Albumentations)") # <-- Updated
    print("Python:", sys.executable)
    print("CUDA:", torch.cuda.is_available(), " BF16:", args.bf16)
    print("="*70, flush=True)

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    # Note: EfficientNet-B5 default is 456. Your size of 384 is a compromise.
    print(f"Using img-size {args.img_size}.")

    train_tfms = A.Compose([
        A.Resize(args.img_size, args.img_size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=int(args.img_size*0.1), max_width=int(args.img_size*0.1), min_holes=1, fill_value=0, p=0.3),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])
    val_tfms = A.Compose([
        A.Resize(args.img_size, args.img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])

    exclude_set = load_exclude_set(args.exclude_dir)
    use_csv = bool(args.train_csv or args.val_csv or args.splits_root)
    if use_csv:
        if args.splits_root:
            if args.fold_index < 1:
                raise ValueError("--fold-index must be set when using --splits-root")
            fold_dir = Path(args.splits_root) / f"fold_{args.fold_index}"
            train_csv = fold_dir / "roi_gt_train.csv"
            val_csv = fold_dir / "roi_gt_val.csv"
        else:
            if not args.train_csv or not args.val_csv:
                raise ValueError("--train-csv and --val-csv are required for CSV mode")
            train_csv = Path(args.train_csv)
            val_csv = Path(args.val_csv)

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        if args.image_col not in train_df.columns or args.label_col not in train_df.columns:
            raise ValueError(f"CSV missing columns: {args.image_col}, {args.label_col}")
        if args.image_col not in val_df.columns or args.label_col not in val_df.columns:
            raise ValueError(f"CSV missing columns: {args.image_col}, {args.label_col}")

        labels_all = pd.concat([train_df[args.label_col], val_df[args.label_col]], ignore_index=True).astype(str)
        classes = sorted(labels_all.unique())
        label_to_idx = {c: i for i, c in enumerate(classes)}
        print("Classes:", classes)
        print("Train CSV:", train_csv)
        print("Val CSV:", val_csv)

        def _build_samples(df):
            samples = []
            for _, row in df.iterrows():
                img_rel = str(row[args.image_col])
                lbl = str(row[args.label_col])
                p = Path(img_rel)
                if not p.is_absolute():
                    if not args.images_root:
                        raise ValueError("Relative image paths require --images-root")
                    p = Path(args.images_root) / img_rel
                samples.append((p, label_to_idx[lbl]))
            return samples

        train_samples = _build_samples(train_df)
        val_samples = _build_samples(val_df)
        assert_no_excluded(train_samples, exclude_set, "Train split")
        assert_no_excluded(val_samples, exclude_set, "Val split")

        if args.ram_cache:
            print(" Using RAM-cached dataset", flush=True)
            train_tfms_ram = A.Compose([t for t in train_tfms if not isinstance(t, A.Resize)])
            val_tfms_ram = A.Compose([t for t in val_tfms if not isinstance(t, A.Resize)])
            train_ds = RAMImageDataset(train_samples, classes, args.img_size, transform=train_tfms_ram)
            val_ds   = RAMImageDataset(val_samples,   classes, args.img_size, transform=val_tfms_ram)
            targets_train = train_ds.targets
        else:
            print(" Using on-disk CSV dataset", flush=True)
            train_ds = CSVImageDataset(train_samples, transform=train_tfms)
            val_ds   = CSVImageDataset(val_samples, transform=val_tfms)
            targets_train = np.array([y for _, y in train_samples], dtype=np.int64)
        val_targets = np.array([y for _, y in val_samples], dtype=np.int64)
    else:
        if not args.crop_root:
            raise ValueError("--crop-root is required when not using CSV mode")

        base = datasets.ImageFolder(root=str(args.crop_root))
        classes = base.classes
        print("Classes:", classes)
        print("Total images:", len(base))

        if args.kfolds and args.fold_index:
            if args.fold_index < 1 or args.fold_index > args.kfolds:
                raise ValueError("--fold-index must be between 1 and --kfolds")
            folds = stratified_kfold_indices(base.targets, args.kfolds, seed=args.seed)
            va_idx = folds[args.fold_index - 1]
            tr_idx = [i for f, fold in enumerate(folds) if f != (args.fold_index - 1) for i in fold]
            print(f"Using stratified kfold: k={args.kfolds}, fold={args.fold_index}")
        else:
            tr_idx, va_idx = simple_split(len(base), val_ratio=args.val_ratio, seed=args.seed)
            print(f"Using random split: val_ratio={args.val_ratio}")
        train_samples = [base.samples[i] for i in tr_idx]
        val_samples   = [base.samples[i] for i in va_idx]
        assert_no_excluded(train_samples, exclude_set, "Train split")
        assert_no_excluded(val_samples, exclude_set, "Val split")

        if args.ram_cache:
            print(" Using RAM-cached dataset", flush=True)
            train_tfms_ram = A.Compose([t for t in train_tfms if not isinstance(t, A.Resize)])
            val_tfms_ram = A.Compose([t for t in val_tfms if not isinstance(t, A.Resize)])
            train_ds = RAMImageDataset(train_samples, classes, args.img_size, transform=train_tfms_ram)
            val_ds   = RAMImageDataset(val_samples,   classes, args.img_size, transform=val_tfms_ram)
            targets_train = train_ds.targets
        else:
            print(" Using on-disk ImageFolder (with Torchvision transforms)", flush=True)
            from torchvision import transforms # Need transforms here
            train_tfms_tv = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                transforms.ToTensor(), transforms.Normalize(imagenet_mean, imagenet_std),
            ])
            val_tfms_tv = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ])
            train_ds = Subset(datasets.ImageFolder(root=str(args.crop_root), transform=train_tfms_tv), tr_idx)
            val_ds   = Subset(datasets.ImageFolder(root=str(args.crop_root), transform=val_tfms_tv), va_idx)
            targets_train = np.array([base.samples[i][1] for i in tr_idx], dtype=np.int64)
        val_targets = np.array([base.samples[i][1] for i in va_idx], dtype=np.int64)

    class_counts = np.bincount(targets_train, minlength=len(classes))
    class_weight = 1.0 / np.maximum(class_counts, 1)
    sample_weight = class_weight[targets_train]
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weight).float(), len(sample_weight), replacement=True)

    print("")
    print_dist("Train", targets_train, classes)
    print_dist("Val", val_targets, classes)

    use_mp = args.num_workers > 0
    dl_kwargs = dict(batch_size=args.batch, num_workers=args.num_workers, pin_memory=True)
    if use_mp:
        dl_kwargs.update(prefetch_factor=2, persistent_workers=False, timeout=120)

    train_loader = DataLoader(train_ds, sampler=sampler, **dl_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    print(f"\nDataLoaders: workers={args.num_workers}")
    print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches", flush=True)

    # --- MODIFIED: Use EfficientNet-B5 ---
    model_name = "EfficientNet-B5"
    model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
    in_f  = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, len(classes))
    backbone = model.features
    # --- END MODIFICATION ---
    
    model = model.to(device).to(memory_format=torch.channels_last)
    print(f"\nUsing {model_name}")

    def set_grad(m, flag):
        for p in m.parameters():
            p.requires_grad_(flag)
    set_grad(backbone, False)
    print(f"Backbone frozen for first {args.warmup_epochs} epochs")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(e):
        if e < args.warmup_epochs:
            return (e + 1) / max(1, args.warmup_epochs)
        t = (e - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = -1.0
    bad = 0
    train_loss_hist, val_loss_hist, val_acc_hist = [], [], []
    start = time.time()

    print("\nStarting training...\n" + "="*70, flush=True)
    accum = max(1, args.accum)

    for ep in range(1, args.epochs + 1):
        if ep == args.warmup_epochs:
            set_grad(model, True)
            print(" Backbone unfrozen\n", flush=True)

        model.train()
        tl = 0.0
        optimizer.zero_grad(set_to_none=True)

        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, ymix, _ = mixup(x, y, args.mixup_alpha)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                out = model(x)
                if ymix is None:
                    loss = criterion(out, y)
                else:
                    y1, y2, lam = ymix
                    loss = lam * criterion(out, y1) + (1 - lam) * criterion(out, y2)
                loss = loss / accum

            if use_cuda:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (bi + 1) % accum == 0:
                if use_cuda:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            tl += (loss.detach() * accum).item() * x.size(0)

        if (len(train_loader) % accum) != 0:
            if use_cuda:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        tl /= len(train_loader.dataset)

        model.eval()
        vl, correct = 0.0, 0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                loss = criterion(out, y)
                vl += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()

        vl /= len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)

        train_loss_hist.append(tl)
        val_loss_hist.append(vl)
        val_acc_hist.append(acc)

        scheduler.step()

        print(f"Ep {ep:03d}/{args.epochs} | Train {tl:.4f} | Val {vl:.4f} | Acc {acc*100:.2f}%", end="")

        if acc > best_acc:
            best_acc = acc
            bad = 0
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "model_name": model_name
            }, args.save_path)
            print("   Best!", flush=True)
        else:
            bad += 1
            print(f"  | No improve: {bad}/{args.patience}", flush=True)
            if bad >= args.patience:
                print("\nEarly stopping.", flush=True)
                break

    elapsed = time.time() - start
    print("="*70 + f"\nDone in {elapsed/60:.1f} min  |  Best Acc: {best_acc*100:.2f}%")

    # plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_hist, label="Train")
    plt.plot(val_loss_hist, label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_acc_hist, label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plot_save_path = Path(args.save_path).parent / "classification_curves.png"
    plt.savefig(plot_save_path, dpi=150)
    print(f"Saved training curves to {plot_save_path}")

if __name__ == "__main__":
    main()
