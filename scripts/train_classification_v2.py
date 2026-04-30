"""
Enhanced Wound Classification Training Script V2 (Fixed)
"""

import os
import sys
import time
import argparse
import random
import json
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
torch.set_num_threads(1)

try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crop-root", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--exp-name", required=True)
    p.add_argument("--backbone", type=str, default="efficientnet_b5", 
                   choices=["convnext_tiny", "convnext_small", "convnext_base", 
                            "efficientnet_v2_m", "efficientnet_b5", "swin_b"])
    p.add_argument("--img-size", type=int, default=456)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--label-smooth", type=float, default=0.05)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--cutmix-alpha", type=float, default=0.0)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--use-focal", action="store_true", default=True)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--swa", dest='swa', action="store_true")
    p.add_argument("--no-swa", dest='swa', action="store_false")
    p.set_defaults(swa=False)
    p.add_argument("--swa-start", type=int, default=75)
    p.add_argument("--swa-lr", type=float, default=1e-5)
    p.add_argument("--tta", dest='tta', action="store_true")
    p.add_argument("--no-tta", dest='tta', action="store_false")
    p.set_defaults(tta=True)
    p.add_argument("--tta-transforms", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ram-cache", action="store_true")
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WoundDataset(Dataset):
    def __init__(self, samples, img_size, is_train=True, ram_cache=False):
        self.samples = samples
        self.img_size = img_size
        self.ram_cache = ram_cache
        
        if is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        self.cached = {}
        if ram_cache:
            print(f"Caching {'train' if is_train else 'val'}...", end="", flush=True)
            for i, (path, _) in enumerate(samples):
                if i % 100 == 0: print(".", end="", flush=True)
                img = cv2.imread(str(path))
                if img is not None:
                    self.cached[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(" done!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if idx in self.cached:
            img = self.cached[idx].copy()
        else:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)['image'], label


def get_model(backbone, num_classes):
    if backbone == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m, "ConvNeXt-Tiny"
    elif backbone == "convnext_small":
        m = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m, "ConvNeXt-Small"
    elif backbone == "convnext_base":
        m = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m, "ConvNeXt-Base"
    elif backbone == "efficientnet_v2_m":
        m = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m, "EfficientNet-V2-M"
    elif backbone == "efficientnet_b5":
        m = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m, "EfficientNet-B5"
    elif backbone == "swin_b":
        m = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m, "Swin-B"


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, num_classes=4):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.alpha = torch.tensor(alpha) if alpha else None
    
    def forward(self, x, y):
        if self.label_smoothing > 0:
            ys = torch.zeros_like(x).scatter_(1, y.unsqueeze(1), 1.0)
            ys = ys * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
            ce = -(ys * F.log_softmax(x, dim=1)).sum(dim=1)
        else:
            ce = F.cross_entropy(x, y, reduction='none')
        pt = torch.exp(-ce)
        fl = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            fl = self.alpha.to(x.device)[y] * fl
        return fl.mean()


def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


@torch.no_grad()
def evaluate_tta(model, samples, img_size, device, n=5):
    model.eval()
    tfms = [
        A.Compose([A.Resize(img_size, img_size), A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1), A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1), A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size), A.RandomRotate90(p=1), A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), ToTensorV2()]),
        A.Compose([A.Resize(int(img_size*1.1), int(img_size*1.1)), A.CenterCrop(img_size, img_size), A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), ToTensorV2()]),
    ][:n]
    
    all_probs, all_labels = [], []
    for path, label in samples:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        prob_sum = None
        for t in tfms:
            x = t(image=img)['image'].unsqueeze(0).to(device)
            p = F.softmax(model(x), dim=1)
            prob_sum = p if prob_sum is None else prob_sum + p
        all_probs.append((prob_sum / len(tfms)).cpu().numpy())
        all_labels.append(label)
    
    probs = np.concatenate(all_probs)
    labels = np.array(all_labels)
    return (probs.argmax(1) == labels).mean(), probs, labels, probs.argmax(1)


def plot_curves(tl, vl, va, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(tl, 'b-', label='Train')
    ax[0].plot(vl, 'r-', label='Val')
    ax[0].legend()
    ax[0].set_title('Loss')
    ax[1].plot([v*100 for v in va], 'g-')
    ax[1].set_title(f'Accuracy (Best: {max(va)*100:.2f}%)')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cm(labels, preds, classes, path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = get_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    exp_dir = Path(args.save_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_path = exp_dir / "best_model.pth"
    print(f"Experiment: {args.exp_name}")
    
    # Data
    crop_root = Path(args.crop_root)
    classes = sorted([d.name for d in crop_root.iterdir() if d.is_dir()])
    print(f"Classes: {classes}")
    
    samples = []
    counts = Counter()
    for ci, cn in enumerate(classes):
        for p in (crop_root / cn).glob("*.*"):
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                samples.append((p, ci))
                counts[cn] += 1
    
    print(f"Total: {len(samples)}")
    for c, n in counts.items():
        print(f"  {c}: {n}")
    
    weights = [len(samples) / (len(classes) * counts[c]) for c in classes]
    
    # Split
    np.random.seed(args.seed)
    by_class = {i: [] for i in range(len(classes))}
    for i, (_, l) in enumerate(samples):
        by_class[l].append(i)
    
    train_idx, val_idx = [], []
    for ci in range(len(classes)):
        idxs = by_class[ci]
        np.random.shuffle(idxs)
        s = int(len(idxs) * 0.8)
        train_idx.extend(idxs[:s])
        val_idx.extend(idxs[s:])
    
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    train_ds = WoundDataset(train_samples, args.img_size, True, args.ram_cache)
    val_ds = WoundDataset(val_samples, args.img_size, False, args.ram_cache)
    
    sampler = WeightedRandomSampler([weights[l] for _, l in train_samples], len(train_samples))
    train_loader = DataLoader(train_ds, args.batch, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Model
    model, model_name = get_model(args.backbone, len(classes))
    model = model.to(device)
    print(f"Model: {model_name}, SWA: {args.swa}, TTA: {args.tta}")
    
    criterion = FocalLoss(weights, args.focal_gamma, args.label_smooth, len(classes)) if args.use_focal else nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epochs, args.min_lr)
    
    swa_model, swa_sched = None, None
    if args.swa:
        swa_model = AveragedModel(model)
        swa_sched = SWALR(optimizer, swa_lr=args.swa_lr)
    
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and not args.bf16)
    
    tl_hist, vl_hist, va_hist = [], [], []
    best_acc, bad = 0.0, 0
    best_probs, best_labels, best_preds = None, None, None
    
    csv_f = open(exp_dir / "log.csv", 'w', newline='')
    csv_w = csv.writer(csv_f)
    csv_w.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'lr', 'best'])
    
    print(f"\nTraining {args.epochs} epochs...")
    print("=" * 50)
    t0 = time.time()
    
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        optimizer.zero_grad()
        
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            if args.mixup_alpha > 0 and np.random.random() < args.mixup_prob:
                x, ya, yb, lam = mixup_data(x, y, args.mixup_alpha)
                mix = True
            else:
                mix = False
            
            with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
                out = model(x)
                loss = (lam * criterion(out, ya) + (1-lam) * criterion(out, yb)) if mix else criterion(out, y)
                loss = loss / args.accum
            
            scaler.scale(loss).backward()
            
            if (bi + 1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            tl += loss.item() * args.accum * x.size(0)
        
        tl /= len(train_ds)
        
        if args.swa and ep >= args.swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            scheduler.step()
        
        # Val
        model.eval()
        vl, cor = 0.0, 0
        probs_l, labels_l, preds_l = [], [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
                    out = model(x)
                    loss = criterion(out, y)
                vl += loss.item() * x.size(0)
                p = F.softmax(out, dim=1)
                pr = out.argmax(1)
                cor += (pr == y).sum().item()
                probs_l.append(p.cpu().numpy())
                labels_l.append(y.cpu().numpy())
                preds_l.append(pr.cpu().numpy())
        
        vl /= len(val_ds)
        va = cor / len(val_ds)
        
        probs_l = np.concatenate(probs_l)
        labels_l = np.concatenate(labels_l)
        preds_l = np.concatenate(preds_l)
        
        tl_hist.append(tl)
        vl_hist.append(vl)
        va_hist.append(va)
        
        lr = optimizer.param_groups[0]['lr']
        best = va > best_acc
        
        print(f"Ep {ep:03d} | TL: {tl:.4f} | VL: {vl:.4f} | Acc: {va*100:.2f}% | LR: {lr:.6f}", end="")
        csv_w.writerow([ep, f"{tl:.6f}", f"{vl:.6f}", f"{va:.6f}", f"{lr:.8f}", "1" if best else "0"])
        csv_f.flush()
        
        if best:
            best_acc = va
            best_probs, best_labels, best_preds = probs_l, labels_l, preds_l
            bad = 0
            torch.save({"model_state": model.state_dict(), "classes": classes, "model_name": model_name,
                       "epoch": ep, "best_acc": best_acc, "config": vars(args)}, save_path)
            print(" * Best!")
        else:
            bad += 1
            print(f" | No imp: {bad}/{args.patience}")
            if bad >= args.patience:
                print(">>> Early stop")
                break
    
    csv_f.close()
    
    if args.swa and swa_model and ep >= args.swa_start:
        print("Updating SWA BN...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save({"model_state": swa_model.module.state_dict(), "classes": classes, "model_name": model_name + " (SWA)"}, exp_dir / "swa_model.pth")
    
    if args.tta:
        print("Running TTA...")
        tta_acc, tta_probs, tta_labels, tta_preds = evaluate_tta(model, val_samples, args.img_size, device, args.tta_transforms)
        print(f"TTA Acc: {tta_acc*100:.2f}%")
        if tta_acc > best_acc:
            best_probs, best_labels, best_preds = tta_probs, tta_labels, tta_preds
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min | Best: {best_acc*100:.2f}%")
    
    plot_curves(tl_hist, vl_hist, va_hist, exp_dir / "curves.png")
    plot_cm(best_labels, best_preds, classes, exp_dir / "confusion.png")
    
    from sklearn.metrics import classification_report
    report = classification_report(best_labels, best_preds, target_names=classes, output_dict=True)
    with open(exp_dir / "report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nReport:")
    for c in classes:
        print(f"  {c}: P={report[c]['precision']:.3f} R={report[c]['recall']:.3f} F1={report[c]['f1-score']:.3f}")
    print(f"  Accuracy: {report['accuracy']:.3f}")
    
    print(f"\n[OK] Saved to {exp_dir}")


if __name__ == "__main__":
    main()