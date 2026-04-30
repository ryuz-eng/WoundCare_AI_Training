"""
Enhanced Wound Classification Training Script V2
Target: 90%+ accuracy with high confidence predictions

Key Improvements:
1. ConvNeXt backbone (better than EfficientNet for medical imaging)
2. Strong augmentation with MixUp + CutMix
3. Label smoothing (0.1) for better calibration
4. Focal Loss to handle class imbalance
5. Test-Time Augmentation (TTA) for evaluation
6. Larger image size (384x384)
7. Longer training with cosine annealing
8. Stochastic Weight Averaging (SWA)

Usage:
    python train_classification_v2.py --crop-root D:/WoundCare/classification_data --save-dir D:/WoundCare/experiments --exp-name C_enhanced_v2

Dataset structure:
    crop-root/
    ├── Stage_1/ (258 images)
    ├── Stage_2/ (361 images)
    ├── Stage_3/ (321 images)
    └── Stage_4/ (305 images)
"""

import os
import sys
import time
import math
import argparse
import random
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Suppress warnings
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
    p = argparse.ArgumentParser(description="Enhanced Wound Classification Training")
    
    # Required
    p.add_argument("--crop-root", required=True, help="Path to cropped images organized by class folders")
    p.add_argument("--save-dir", required=True, help="Directory to save experiment outputs")
    p.add_argument("--exp-name", required=True, help="Experiment name")
    
    # Model - ConvNeXt is better for medical imaging
    p.add_argument("--backbone", type=str, default="convnext_base", 
                   choices=["convnext_tiny", "convnext_small", "convnext_base", 
                            "efficientnet_v2_m", "efficientnet_b5", "swin_b"],
                   help="Backbone architecture")
    
    # Training
    p.add_argument("--img-size", type=int, default=384, help="Image size (larger = better)")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--epochs", type=int, default=150, help="Max epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for cosine annealing")
    p.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    p.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    
    p.add_argument("--monitor", type=str, default="loss", choices=["loss", "acc"],
                   help="Metric to monitor for early stopping / best_model (loss or acc)")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="Minimum change to qualify as an improvement for the monitored metric")

    # LR scheduler
    p.add_argument("--sched", type=str, default="plateau", choices=["cosine", "plateau"],
                   help="LR scheduler: cosine annealing or ReduceLROnPlateau")
    p.add_argument("--plateau-factor", type=float, default=0.1, help="ReduceLROnPlateau factor")
    p.add_argument("--plateau-patience", type=int, default=4, help="ReduceLROnPlateau patience (epochs)")

    # Advanced techniques
    p.add_argument("--label-smooth", type=float, default=0.1, help="Label smoothing factor")
    p.add_argument("--mixup-alpha", type=float, default=0.4, help="MixUp alpha (0 to disable)")
    p.add_argument("--cutmix-alpha", type=float, default=1.0, help="CutMix alpha (0 to disable)")
    p.add_argument("--mixup-prob", type=float, default=0.5, help="Probability of applying MixUp/CutMix")
    p.add_argument("--drop-path", type=float, default=0.2, help="Drop path rate")
    p.add_argument("--sampler", type=str, default="weighted", choices=["weighted", "shuffle"],
                   help="Training sampler: weighted (balances classes) or shuffle")
    p.add_argument("--use-focal", action="store_true", help="Use Focal Loss")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    
    # SWA
    p.add_argument("--swa", action="store_true", help="Use Stochastic Weight Averaging")
    p.add_argument("--swa-start", type=int, default=100, help="Start SWA at this epoch")
    p.add_argument("--swa-lr", type=float, default=1e-5, help="SWA learning rate")
    
    # TTA
    p.add_argument("--tta", action="store_true", help="Use Test-Time Augmentation")
    p.add_argument("--tta-transforms", type=int, default=5, help="Number of TTA transforms")
    
    # Performance
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--ram-cache", action="store_true", help="Cache images in RAM")
    p.add_argument("--accum", type=int, default=2, help="Gradient accumulation steps")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16")
    
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class WoundDataset(Dataset):
    """Enhanced dataset with strong augmentations"""
    
    def __init__(self, samples, img_size, is_train=True, ram_cache=False):
        self.samples = samples
        self.img_size = img_size
        self.is_train = is_train
        self.ram_cache = ram_cache
        
        # Strong augmentation for training
        if is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
                    A.GridDistortion(p=1.0),
                    A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=4.0, p=1.0),
                    A.Sharpen(p=1.0),
                    A.Emboss(p=1.0),
                ], p=0.2),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                                min_holes=1, min_height=8, min_width=8, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        # RAM caching
        self.cached_images = {}
        if ram_cache:
            print(f"Caching {'train' if is_train else 'val'} images in RAM...", end="", flush=True)
            for i, (path, _) in enumerate(samples):
                if i % 100 == 0:
                    print(".", end="", flush=True)
                self.cached_images[i] = self._load_image(path)
            print(" done!")
    
    def _load_image(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        if idx in self.cached_images:
            img = self.cached_images[idx].copy()
        else:
            img = self._load_image(path)
        
        augmented = self.transform(image=img)
        return augmented['image'], label


def get_tta_transforms(img_size):
    """Get TTA transforms"""
    return [
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(img_size, img_size),
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(int(img_size * 1.1), int(img_size * 1.1)),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]


# ============================================================
# MODEL
# ============================================================

def get_model(backbone, num_classes, drop_path=0.2):
    """Get model based on backbone choice"""
    
    if backbone == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        name = "ConvNeXt-Tiny"
    
    elif backbone == "convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        name = "ConvNeXt-Small"
    
    elif backbone == "convnext_base":
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        name = "ConvNeXt-Base"
    
    elif backbone == "efficientnet_v2_m":
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        name = "EfficientNet-V2-M"
    
    elif backbone == "efficientnet_b5":
        model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        name = "EfficientNet-B5"
    
    elif backbone == "swin_b":
        model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
        name = "Swin-B"
    
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    return model, name


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, num_classes=4):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_smooth = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            targets_smooth = targets_smooth * (1 - self.label_smoothing) + \
                           self.label_smoothing / self.num_classes
            ce_loss = -(targets_smooth * F.log_softmax(inputs, dim=1)).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


# ============================================================
# MIXUP / CUTMIX
# ============================================================

def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    # Get bounding box
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    x_cut = x.clone()
    x_cut[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return x_cut, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for MixUp/CutMix"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# EVALUATION WITH TTA
# ============================================================

@torch.no_grad()
def evaluate_with_tta(model, val_samples, img_size, device, num_tta=5):
    """Evaluate with Test-Time Augmentation"""
    model.eval()
    
    tta_transforms = get_tta_transforms(img_size)[:num_tta]
    
    all_probs = []
    all_labels = []
    
    for path, label in val_samples:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Accumulate predictions from TTA
        probs_sum = None
        
        for transform in tta_transforms:
            augmented = transform(image=img)
            x = augmented['image'].unsqueeze(0).to(device)
            
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            
            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum += probs
        
        # Average predictions
        probs_avg = probs_sum / len(tta_transforms)
        
        all_probs.append(probs_avg.cpu().numpy())
        all_labels.append(label)
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    
    preds = np.argmax(all_probs, axis=1)
    accuracy = (preds == all_labels).mean()
    
    return accuracy, all_probs, all_labels, preds


# ============================================================
# METRICS AND VISUALIZATION
# ============================================================

def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def plot_training_curves(train_loss, val_loss, val_acc, save_path):
    """Plot training curves"""
    epochs = range(1, len(train_loss) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, [a * 100 for a in val_acc], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Validation Accuracy (Best: {max(val_acc)*100:.2f}%)', fontsize=14)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(labels, preds, classes, save_path):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix as cm_func
    
    cm = cm_func(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    thresh = cm_normalized.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)',
                   ha='center', va='center', fontsize=10,
                   color='white' if cm_normalized[i, j] > thresh else 'black')
    
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return cm


def plot_reliability_diagram(probs, labels, save_path, n_bins=15):
    """Plot reliability diagram"""
    ece = compute_ece(probs, labels, n_bins)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    bin_accs = []
    bin_confs = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_accs.append(accuracies[in_bin].mean())
            bin_confs.append(confidences[in_bin].mean())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_lower + bin_upper) / 2)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bin_centers = (bin_lowers + bin_uppers) / 2
    bin_width = 1.0 / n_bins
    ax.bar(bin_centers, bin_accs, width=bin_width * 0.8, alpha=0.7, color='steelblue', edgecolor='black')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram (ECE={ece:.4f})', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return ece


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    args = get_args()
    set_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup experiment directory
    exp_dir = Path(args.save_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_path = exp_dir / "best_model.pth"
    best_loss_path = exp_dir / "best_loss.pth"
    best_acc_path = exp_dir / "best_acc.pth"
    print(f"\nExperiment: {args.exp_name}")
    print(f"Output dir: {exp_dir}")
    
    # Load dataset
    crop_root = Path(args.crop_root)
    classes = sorted([d.name for d in crop_root.iterdir() if d.is_dir()])
    print(f"\nClasses: {classes}")
    
    # Collect all samples
    all_samples = []
    class_counts = Counter()
    
    for class_idx, class_name in enumerate(classes):
        class_dir = crop_root / class_name
        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                all_samples.append((img_path, class_idx))
                class_counts[class_name] += 1
    
    print(f"Total samples: {len(all_samples)}")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Calculate class weights for balanced sampling
    class_weights = []
    for class_name in classes:
        weight = len(all_samples) / (len(classes) * class_counts[class_name])
        class_weights.append(weight)
    print(f"\nClass weights: {[f'{w:.3f}' for w in class_weights]}")
    
    # Split dataset (stratified)
    np.random.seed(args.seed)
    indices_by_class = {i: [] for i in range(len(classes))}
    for idx, (_, label) in enumerate(all_samples):
        indices_by_class[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    for class_idx in range(len(classes)):
        class_indices = indices_by_class[class_idx]
        np.random.shuffle(class_indices)
        split = int(len(class_indices) * 0.8)
        train_indices.extend(class_indices[:split])
        val_indices.extend(class_indices[split:])
    
    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]
    
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Create datasets
    train_ds = WoundDataset(train_samples, args.img_size, is_train=True, ram_cache=args.ram_cache)
    val_ds = WoundDataset(val_samples, args.img_size, is_train=False, ram_cache=args.ram_cache)
    
    # Sampler / shuffling strategy
    sampler = None
    shuffle_train = True
    if args.sampler == "weighted":
        # Weighted sampler for class balance
        sample_weights = [class_weights[label] for _, label in train_samples]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle_train = False
        if args.use_focal:
            print("Warning: You enabled both weighted sampling and focal loss. If training is unstable, try --sampler shuffle or disable focal.")

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, sampler=sampler, shuffle=shuffle_train,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Model
    model, model_name = get_model(args.backbone, len(classes), args.drop_path)
    model = model.to(device)
    print(f"\nModel: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    if args.use_focal:
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smooth,
            num_classes=len(classes)
        )
        print(f"Loss: Focal Loss (gamma={args.focal_gamma}, label_smooth={args.label_smooth})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
        print(f"Loss: CrossEntropy (label_smooth={args.label_smooth})")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    if args.sched == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.plateau_factor, patience=args.plateau_patience,
            min_lr=args.min_lr, verbose=True
        )

    # SWA
    swa_model = None
    swa_scheduler = None
    if args.swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
        print(f"SWA enabled (starting at epoch {args.swa_start})")
    
    # Mixed precision
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and not args.bf16)
    
    # Training history
    train_loss_hist = []
    val_loss_hist = []
    val_acc_hist = []

    # Track BOTH: best-by-loss and best-by-accuracy
    best_val_loss = float("inf")
    best_loss_epoch = -1
    best_loss_probs = None
    best_loss_labels = None
    best_loss_preds = None

    best_acc = 0.0
    best_acc_epoch = -1
    best_acc_probs = None
    best_acc_labels = None
    best_acc_preds = None

    # Best model (based on args.monitor) for reports
    best_probs = None
    best_labels = None
    best_preds = None

    # Early stopping based on args.monitor
    bad_epochs = 0

    # Config for saving
    config = {
        "backbone": args.backbone,
        "img_size": args.img_size,
        "batch": args.batch,
        "lr": args.lr,
        "label_smooth": args.label_smooth,
        "mixup_alpha": args.mixup_alpha,
        "cutmix_alpha": args.cutmix_alpha,
        "use_focal": args.use_focal,
        "seed": args.seed,
    }
    
    # CSV logging
    csv_path = exp_dir / "training_log.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'lr', 'is_best_model', 'is_best_loss', 'is_best_acc'])
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Apply MixUp / CutMix (use a single draw so the total probability is correct)
            use_mixup = False
            use_cutmix = False
            if (args.mixup_alpha > 0 or args.cutmix_alpha > 0) and args.mixup_prob > 0:
                r = np.random.random()
                if r < args.mixup_prob:
                    # Choose which augmentation to apply
                    if args.mixup_alpha > 0 and args.cutmix_alpha > 0:
                        if np.random.random() < 0.5:
                            use_mixup = True
                        else:
                            use_cutmix = True
                    elif args.mixup_alpha > 0:
                        use_mixup = True
                    elif args.cutmix_alpha > 0:
                        use_cutmix = True
            
            if use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, args.mixup_alpha)
            elif use_cutmix:
                x, y_a, y_b, lam = cutmix_data(x, y, args.cutmix_alpha)
            
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                
                if use_mixup or use_cutmix:
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, y)
                
                loss = loss / args.accum
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * args.accum * x.size(0)
        
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        all_probs = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                
                val_loss += loss.item() * x.size(0)
                
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        
        # Record history
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        
        cur_lr = optimizer.param_groups[0]['lr']

        # Determine improvements
        is_best_loss = val_loss < (best_val_loss - args.min_delta)
        # For accuracy, min-delta should be tiny (accuracy is in [0,1])
        acc_min_delta = max(args.min_delta, 1e-6)
        is_best_acc = val_acc > (best_acc + acc_min_delta)

        if args.monitor == "loss":
            is_best_model = is_best_loss
        else:
            is_best_model = is_best_acc

        print(f"Ep {epoch:03d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"Acc: {val_acc*100:.2f}% | "
              f"LR: {cur_lr:.6f}", end="")

        # Log to CSV
        csv_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{val_acc:.6f}", f"{cur_lr:.8f}",
            "1" if is_best_model else "0",
            "1" if is_best_loss else "0",
            "1" if is_best_acc else "0",
        ])
        csv_file.flush()

        # Save best-by-loss checkpoint
        if is_best_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            best_loss_probs = all_probs
            best_loss_labels = all_labels
            best_loss_preds = all_preds

            save_dict = {
                "model_state": model.state_dict(),
                "classes": classes,
                "model_name": model_name,
                "epoch": epoch,
                "best_val_loss": float(best_val_loss),
                "val_acc_at_best_loss": float(val_acc),
                "config": config,
            }
            if args.swa and swa_model is not None and epoch >= args.swa_start:
                save_dict["swa_model_state"] = swa_model.module.state_dict()

            torch.save(save_dict, best_loss_path)

        # Save best-by-accuracy checkpoint
        if is_best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch
            best_acc_probs = all_probs
            best_acc_labels = all_labels
            best_acc_preds = all_preds

            save_dict = {
                "model_state": model.state_dict(),
                "classes": classes,
                "model_name": model_name,
                "epoch": epoch,
                "best_acc": float(best_acc),
                "val_loss_at_best_acc": float(val_loss),
                "config": config,
            }
            if args.swa and swa_model is not None and epoch >= args.swa_start:
                save_dict["swa_model_state"] = swa_model.module.state_dict()

            torch.save(save_dict, best_acc_path)

        # Best model for reports + early stopping (based on args.monitor)
        if is_best_model:
            bad_epochs = 0
            best_probs = all_probs
            best_labels = all_labels
            best_preds = all_preds

            save_dict = {
                "model_state": model.state_dict(),
                "classes": classes,
                "model_name": model_name,
                "epoch": epoch,
                "monitor": args.monitor,
                "best_val_loss": float(best_val_loss),
                "best_acc": float(best_acc),
                "config": config,
            }
            if args.swa and swa_model is not None and epoch >= args.swa_start:
                save_dict["swa_model_state"] = swa_model.module.state_dict()

            torch.save(save_dict, save_path)
            print("Best!", flush=True)
        else:
            bad_epochs += 1
            print(f" | No improve: {bad_epochs}/{args.patience}", flush=True)

            if bad_epochs >= args.patience:
                print("\n>>> Early stopping triggered.")
                break

        # LR scheduling (after validation)
        if args.swa and swa_model is not None and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            if args.sched == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
    csv_file.close()
    
    # Update SWA batch normalization
    if args.swa and swa_model is not None:
        print("\nUpdating SWA batch normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        
        # Save SWA model separately
        swa_save_path = exp_dir / "swa_model.pth"
        torch.save({
            "model_state": swa_model.module.state_dict(),
            "classes": classes,
            "model_name": model_name + " (SWA)",
            "config": config,
        }, swa_save_path)
        print(f"SWA model saved to: {swa_save_path}")
    
    # Evaluate with TTA
    if args.tta:
        # Load best_model weights before running TTA (so TTA is evaluated on your best checkpoint)
        if save_path.exists():
            ckpt = torch.load(save_path, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=True)
        print("\nEvaluating with Test-Time Augmentation...")
        tta_acc, tta_probs, tta_labels, tta_preds = evaluate_with_tta(
            model, val_samples, args.img_size, device, args.tta_transforms
        )
        print(f"TTA Accuracy: {tta_acc * 100:.2f}%")
        
        if tta_acc > best_acc:
            best_probs = tta_probs
            best_labels = tta_labels
            best_preds = tta_preds
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"Training completed in {elapsed/60:.1f} minutes")
    print(f"Monitor metric: {args.monitor} (min_delta={args.min_delta})")
    print(f"Best Val Loss: {best_val_loss:.4f} (epoch {best_loss_epoch})")
    print(f"Best Val Acc : {best_acc * 100:.2f}% (epoch {best_acc_epoch})")
    print("=" * 70)
    
    # Generate visualizations
    print("\nGenerating reports...")
    
    # Training curves
    plot_training_curves(train_loss_hist, val_loss_hist, val_acc_hist,
                        exp_dir / "training_curves.png")
    print("  Saved: training_curves.png")
    
    # Confusion matrix
    plot_confusion_matrix(best_labels, best_preds, classes,
                         exp_dir / "confusion_matrix.png")
    print("  Saved: confusion_matrix.png")
    
    # Reliability diagram
    ece = plot_reliability_diagram(best_probs, best_labels,
                                   exp_dir / "reliability_diagram.png")
    print(f"  Saved: reliability_diagram.png (ECE: {ece:.4f})")
    
    # Classification report
    from sklearn.metrics import classification_report
    report = classification_report(best_labels, best_preds, target_names=classes, output_dict=True)
    
    with open(exp_dir / "classification_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    print("  Saved: classification_report.json")
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for class_name in classes:
        metrics = report[class_name]
        print(f"{class_name:<15} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} "
              f"{metrics['f1-score']:<12.3f} {int(metrics['support']):<10}")
    print("-" * 60)
    print(f"{'Accuracy':<15} {'':<12} {'':<12} {report['accuracy']:<12.3f} {int(report['macro avg']['support']):<10}")
    print(f"{'Macro Avg':<15} {report['macro avg']['precision']:<12.3f} {report['macro avg']['recall']:<12.3f} "
          f"{report['macro avg']['f1-score']:<12.3f}")
    
    # Summary
    summary = {
        "experiment": args.exp_name,
        "model": model_name,
        "best_accuracy": float(best_acc),
        "ece": float(ece),
        "epochs_trained": len(train_loss_hist),
        "training_time_min": elapsed / 60,
        "config": config,
    }
    
    with open(exp_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print("  Saved: summary.json")
    
    print(f"\n All outputs saved to: {exp_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
