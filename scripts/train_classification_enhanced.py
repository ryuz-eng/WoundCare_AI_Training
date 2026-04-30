"""
Wound Classification Training Script (Enhanced for FYP Report)
Run: python -u train_classification_enhanced.py --crop-root ... --save-dir ... --exp-name C1_baseline

Features:
  - CSV logging of all metrics per epoch
  - Training curves (loss + accuracy)
  - Confusion matrix
  - Classification report (precision, recall, F1)
  - Reliability diagram (ECE)
  - Backbone selection (B3, B4, B5)
  - Experiment naming for organized outputs

Key flags:
  --exp-name       : experiment name (e.g., C1_baseline, C2_highres)
  --backbone       : efficientnet version (b3, b4, b5)
  --ram-cache      : preload images into RAM for faster epochs
  --accum N        : gradient accumulation steps
  --bf16           : use bfloat16 autocast (RTX 40-series)
"""

import os, sys, time, math, argparse, random, json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import datasets, models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import csv

# System/thread knobs
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def get_args():
    p = argparse.ArgumentParser()
    # Required
    p.add_argument("--crop-root", required=True, help="Path to cropped images organized by class folders")
    p.add_argument("--save-path", required=True, help="Path to save best model (e.g., D:/WoundCare/experiments/C1_baseline/best_model.pth)")
    
    # Model
    p.add_argument("--backbone", type=str, default="b5", choices=["b3", "b4", "b5"],
                   help="EfficientNet backbone version")
    
    # Training
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--label-smooth", type=float, default=0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--mixup-alpha", type=float, default=0)
    p.add_argument("--seed", type=int, default=42)
    
    # Performance
    p.add_argument("--ram-cache", action="store_true", help="Preload images into RAM")
    p.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast")
    
    return p.parse_args()


def simple_split(n, val_ratio=0.3, seed=42):
    """Simple train/val split without sklearn"""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1 - val_ratio))
    return idx[:cut].tolist(), idx[cut:].tolist()


class RAMImageDataset(Dataset):
    """RAM-cached dataset for faster training"""
    def __init__(self, samples, classes, img_size, transform):
        self.samples = samples
        self.classes = classes
        self.H = self.W = img_size
        self.transform = transform
        self.data = []
        self.targets = []
        print(f"RAM Caching {len(samples)} images:", end="", flush=True)
        st = time.time()
        for i, (p, y) in enumerate(samples):
            if i % 100 == 0:
                print(".", end="", flush=True)
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


def mixup(x, y, alpha):
    """Mixup data augmentation"""
    if alpha <= 0:
        return x, None, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1 - lam) * x[idx]
    return x, (y, y[idx], lam), idx


def get_model(backbone, num_classes):
    """Get EfficientNet model based on backbone choice"""
    if backbone == "b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        model_name = "EfficientNet-B3"
    elif backbone == "b4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        model_name = "EfficientNet-B4"
    else:  # b5
        model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
        model_name = "EfficientNet-B5"
    
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    return model, model_name


def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    return ece, bin_accs, bin_confs, bin_counts, bin_lowers, bin_uppers


def plot_reliability_diagram(probs, labels, save_path, n_bins=15):
    """Plot reliability diagram with ECE"""
    ece, bin_accs, bin_confs, bin_counts, bin_lowers, bin_uppers = compute_ece(probs, labels, n_bins)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot bars
    bin_centers = (bin_lowers + bin_uppers) / 2
    bin_width = 1.0 / n_bins
    ax.bar(bin_centers, bin_accs, width=bin_width * 0.8, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'b--', linewidth=2, label='Perfect calibration')
    
    # Plot actual calibration line
    valid_bins = [i for i, c in enumerate(bin_counts) if c > 0]
    if valid_bins:
        ax.plot([bin_confs[i] for i in valid_bins], [bin_accs[i] for i in valid_bins], 
                'o-', color='darkorange', linewidth=2, markersize=6, label='Model calibration')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram (ECE={ece:.3f})', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return ece


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Plot and save confusion matrix"""
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return cm


def compute_classification_report(y_true, y_pred, classes):
    """Compute precision, recall, F1 for each class"""
    n_classes = len(classes)
    report = {}
    
    for i, cls in enumerate(classes):
        tp = sum((t == i and p == i) for t, p in zip(y_true, y_pred))
        fp = sum((t != i and p == i) for t, p in zip(y_true, y_pred))
        fn = sum((t == i and p != i) for t, p in zip(y_true, y_pred))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(t == i for t in y_true)
        
        report[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
    
    # Overall metrics
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_precision = np.mean([report[c]['precision'] for c in classes])
    macro_recall = np.mean([report[c]['recall'] for c in classes])
    macro_f1 = np.mean([report[c]['f1-score'] for c in classes])
    
    report['accuracy'] = accuracy
    report['macro avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1-score': macro_f1,
        'support': len(y_true)
    }
    
    return report


def print_classification_report(report, classes):
    """Print formatted classification report"""
    print("\n" + "="*60)
    print(f"Overall accuracy: {report['accuracy']*100:.4f} %")
    print("\nPer-class report:")
    print(f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print()
    for cls in classes:
        r = report[cls]
        print(f"{cls:>12} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1-score']:>10.3f} {r['support']:>10}")
    print()
    print(f"{'accuracy':>12} {'':>10} {'':>10} {report['accuracy']:>10.3f} {report['macro avg']['support']:>10}")
    r = report['macro avg']
    print(f"{'macro avg':>12} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1-score']:>10.3f} {r['support']:>10}")
    print("="*60)


def plot_training_curves(train_loss, val_loss, val_acc, save_path):
    """Plot and save training curves"""
    epochs = range(1, len(train_loss) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, [a*100 for a in val_acc], 'g-', label='Val Accuracy', linewidth=2)
    best_epoch = np.argmax(val_acc) + 1
    best_acc = max(val_acc) * 100
    ax2.scatter([best_epoch], [best_acc], color='red', s=100, zorder=5, label=f'Best: {best_acc:.2f}%')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = get_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create output directory (use parent of save_path)
    save_path = Path(args.save_path)
    exp_dir = save_path.parent
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_name = exp_dir.name  # Use folder name as experiment name
    
    print("="*70)
    print(f"WOUND CLASSIFICATION TRAINING - {exp_name}")
    print(f"Output directory: {exp_dir}")
    print(f"Backbone: EfficientNet-{args.backbone.upper()}")
    print(f"Image size: {args.img_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch}")
    print(f"MixUp alpha: {args.mixup_alpha}")
    print(f"Device: {device}")
    print(f"AMP dtype: {amp_dtype}")
    print("="*70, flush=True)
    
    # Save experiment config
    config = vars(args)
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config['exp_name'] = exp_name
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Transforms
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    
    train_tfms = A.Compose([
        A.Resize(args.img_size, args.img_size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=int(args.img_size*0.1), max_width=int(args.img_size*0.1), 
                        min_holes=1, fill_value=0, p=0.3),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])
    val_tfms = A.Compose([
        A.Resize(args.img_size, args.img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])
    
    # Load dataset
    base = datasets.ImageFolder(root=str(args.crop_root))
    classes = base.classes
    print(f"\nClasses: {classes}")
    print(f"Total images: {len(base)}")
    
    tr_idx, va_idx = simple_split(len(base), val_ratio=0.3, seed=args.seed)
    train_samples = [base.samples[i] for i in tr_idx]
    val_samples = [base.samples[i] for i in va_idx]
    
    if args.ram_cache:
        print("\nUsing RAM-cached dataset", flush=True)
        train_tfms_ram = A.Compose([t for t in train_tfms if not isinstance(t, A.Resize)])
        val_tfms_ram = A.Compose([t for t in val_tfms if not isinstance(t, A.Resize)])
        train_ds = RAMImageDataset(train_samples, classes, args.img_size, transform=train_tfms_ram)
        val_ds = RAMImageDataset(val_samples, classes, args.img_size, transform=val_tfms_ram)
        targets_train = train_ds.targets
    else:
        print("\nUsing on-disk ImageFolder", flush=True)
        from torchvision import transforms
        train_tfms_tv = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        val_tfms_tv = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        train_ds = Subset(datasets.ImageFolder(root=str(args.crop_root), transform=train_tfms_tv), tr_idx)
        val_ds = Subset(datasets.ImageFolder(root=str(args.crop_root), transform=val_tfms_tv), va_idx)
        targets_train = np.array([base.samples[i][1] for i in tr_idx], dtype=np.int64)
    
    # Class weights for imbalanced data
    class_counts = np.bincount(targets_train, minlength=len(classes))
    class_weight = 1.0 / np.maximum(class_counts, 1)
    sample_weight = class_weight[targets_train]
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weight).float(), len(sample_weight), replacement=True)
    
    print("\nClass distribution (train):")
    for i, c in enumerate(classes):
        print(f"  {c}: {class_counts[i]}")
    
    # DataLoaders
    dl_kwargs = dict(batch_size=args.batch, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, sampler=sampler, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    
    print(f"\nTrain: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches", flush=True)
    
    # Model
    model, model_name = get_model(args.backbone, len(classes))
    model = model.to(device).to(memory_format=torch.channels_last)
    print(f"\nUsing {model_name}")
    
    # Freeze backbone initially
    def set_grad(m, flag):
        for p in m.parameters():
            p.requires_grad_(flag)
    set_grad(model.features, False)
    print(f"Backbone frozen for first {args.warmup_epochs} epochs")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    def lr_lambda(e):
        if e < args.warmup_epochs:
            return (e + 1) / max(1, args.warmup_epochs)
        t = (e - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # CSV logging
    csv_path = exp_dir / "training_log.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'learning_rate', 'best'])
    
    # Training loop
    best_acc = -1.0
    bad = 0
    train_loss_hist, val_loss_hist, val_acc_hist = [], [], []
    start = time.time()
    accum = max(1, args.accum)
    
    print("\nStarting training...\n" + "="*70, flush=True)
    
    for ep in range(1, args.epochs + 1):
        if ep == args.warmup_epochs:
            set_grad(model, True)
            print(">>> Backbone unfrozen\n", flush=True)
        
        # Training
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
        
        # Validation
        model.eval()
        vl, correct = 0.0, 0
        all_probs, all_labels, all_preds = [], [], []
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                loss = criterion(out, y)
                vl += loss.item() * x.size(0)
                
                probs = F.softmax(out, dim=1)
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        vl /= len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)
        
        train_loss_hist.append(tl)
        val_loss_hist.append(vl)
        val_acc_hist.append(acc)
        
        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        is_best = acc > best_acc
        print(f"Ep {ep:03d}/{args.epochs} | Train {tl:.4f} | Val {vl:.4f} | Acc {acc*100:.2f}%", end="")
        
        # Log to CSV
        csv_writer.writerow([ep, f"{tl:.6f}", f"{vl:.6f}", f"{acc:.6f}", f"{cur_lr:.8f}", "1" if is_best else "0"])
        csv_file.flush()
        
        if is_best:
            best_acc = acc
            bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "model_name": model_name,
                "epoch": ep,
                "best_acc": best_acc,
                "config": config
            }, save_path)
            
            # Save best validation predictions for final metrics
            best_probs = np.concatenate(all_probs, axis=0)
            best_labels = np.concatenate(all_labels, axis=0)
            best_preds = np.concatenate(all_preds, axis=0)
            
            print("   Best!", flush=True)
        else:
            bad += 1
            print(f"  | No improve: {bad}/{args.patience}", flush=True)
            if bad >= args.patience:
                print("\n>>> Early stopping.", flush=True)
                break
    
    csv_file.close()
    elapsed = time.time() - start
    
    print("="*70)
    print(f"Training completed in {elapsed/60:.1f} min")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    print("="*70)
    
    # Generate all plots and reports
    print("\nGenerating reports and visualizations...")
    
    # 1. Training curves
    plot_training_curves(train_loss_hist, val_loss_hist, val_acc_hist, 
                         exp_dir / "training_curves.png")
    print(f"  Saved: training_curves.png")
    
    # 2. Confusion matrix
    cm = plot_confusion_matrix(best_labels, best_preds, classes, 
                               exp_dir / "confusion_matrix.png")
    print(f"  Saved: confusion_matrix.png")
    
    # 3. Classification report
    report = compute_classification_report(best_labels, best_preds, classes)
    print_classification_report(report, classes)
    
    # Save report as JSON
    with open(exp_dir / "classification_report.json", 'w') as f:
        # Convert numpy types to Python types
        report_serializable = {}
        for k, v in report.items():
            if isinstance(v, dict):
                report_serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else int(vv) 
                                          for kk, vv in v.items()}
            else:
                report_serializable[k] = float(v)
        json.dump(report_serializable, f, indent=2)
    print(f"  Saved: classification_report.json")
    
    # 4. Reliability diagram (ECE)
    ece = plot_reliability_diagram(best_probs, best_labels, 
                                   exp_dir / "reliability_diagram.png")
    print(f"  Saved: reliability_diagram.png")
    print(f"  ECE (15-bin): {ece:.4f}")
    
    # 5. Summary
    summary = {
        "experiment": exp_name,
        "model": model_name,
        "best_accuracy": float(best_acc),
        "ece": float(ece),
        "epochs_trained": len(train_loss_hist),
        "training_time_min": elapsed / 60,
        "final_train_loss": float(train_loss_hist[-1]),
        "final_val_loss": float(val_loss_hist[-1]),
        "config": config
    }
    with open(exp_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: summary.json")
    
    print(f"\n All outputs saved to: {exp_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
