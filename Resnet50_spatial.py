# ============================================================
# ResNet50 - 6 Class Real-vs-Fake Training Script (Kaggle GPU)
# ============================================================
# Dataset structure:
# train_dirs = {"real": ".../train/real", "deepfake": "...", ...}
# valid_dirs = {"real": ".../valid/valid real", "deepfake": "...", ...}
#
# Metrics: Accuracy, Precision, Recall, F1 (macro + per-class), ROC-AUC, Kappa, MCC
# Outputs: Model, plots, metrics.json, confusion matrices, ROC curves, per-class analysis
# ============================================================

import os
import random
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, cohen_kappa_score, matthews_corrcoef,
    classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import cycle

# ---------------- USER CONFIG ----------------
OUT_DIR = Path("/kaggle/working/resnet50_output")
NUM_CLASSES = 6  # REMOVED FLUX1
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 4
USE_AMP = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SAVE_MODEL_PATH = OUT_DIR / "resnet50_6class.pth"

# --- REMOVED FLUX1 FROM DATASET ---
train_dirs = {
    "real": r"/kaggle/input/real-vs-fake/model/train/real",
    "deepfake": r"/kaggle/input/real-vs-fake/model/train/deepfake",
    "stylegan": r"/kaggle/input/real-vs-fake/model/train/stylegan",
    "stylegan2": r"/kaggle/input/real-vs-fake/model/train/stylegan2",
    "pggan": r"/kaggle/input/real-vs-fake/model/train/pggan",
    "stablediffusion": r"/kaggle/input/real-vs-fake/model/train/stable diffusion",
}
valid_dirs = {
    "real": r"/kaggle/input/real-vs-fake/model/valid/valid real",
    "deepfake": r"/kaggle/input/real-vs-fake/model/valid/valid deepfake",
    "stylegan": r"/kaggle/input/real-vs-fake/model/valid/valid style gan",
    "stylegan2": r"/kaggle/input/real-vs-fake/model/valid/valid style gan 2",
    "pggan": r"/kaggle/input/real-vs-fake/model/valid/valid pggan",
    "stablediffusion": r"/kaggle/input/real-vs-fake/model/valid/valid stable diffusion",
}
# ------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- reproducibility ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ---------------- Custom Dataset Loader ----------------
class CustomImageDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def load_custom_split(dir_dict, transform):
    data, labels = [], []
    class_names = list(dir_dict.keys())
    class_counts = {}
    for label_idx, class_name in enumerate(class_names):
        folder = Path(dir_dict[class_name])
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        count = 0
        for img_file in folder.glob("*.*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                data.append(str(img_file))
                labels.append(label_idx)
                count += 1
        class_counts[class_name] = count
    return data, labels, class_names, class_counts

# ---------------- Transforms ----------------
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- Load Data ----------------
train_data, train_labels, class_names, train_counts = load_custom_split(train_dirs, train_transforms)
val_data, val_labels, _, val_counts = load_custom_split(valid_dirs, val_transforms)

train_dataset = CustomImageDataset(train_data, train_labels, train_transforms)
val_dataset = CustomImageDataset(val_data, val_labels, val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation images.")
print("Classes:", class_names)
print("\nClass Distribution (Train):", train_counts)
print("Class Distribution (Val):", val_counts)

# Plot class distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(train_counts.keys(), train_counts.values())
ax1.set_title("Training Set Distribution")
ax1.set_ylabel("Number of Images")
ax1.tick_params(axis='x', rotation=45)
ax2.bar(val_counts.keys(), val_counts.values())
ax2.set_title("Validation Set Distribution")
ax2.set_ylabel("Number of Images")
ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "class_distribution.png", dpi=150)
plt.close()

# ---------------- Model ----------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# ---------------- Metrics ----------------
def evaluate_model(model, loader):
    model.eval()
    preds, probs, targets = [], [], []
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(imgs)
                loss = criterion(out, labels)
            running_loss += loss.item() * imgs.size(0)
            soft = F.softmax(out, dim=1).cpu().numpy()
            pred_labels = np.argmax(soft, axis=1)
            preds.extend(pred_labels.tolist())
            probs.extend(soft.tolist())
            targets.extend(labels.cpu().numpy().tolist())
            correct += (pred_labels == labels.cpu().numpy()).sum()
            total += labels.size(0)
    avg_loss = running_loss / len(loader.dataset)
    avg_acc = correct / total
    return np.array(preds), np.array(probs), np.array(targets), avg_loss, avg_acc

def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        "accuracy": acc, 
        "precision": prec, 
        "recall": rec, 
        "f1": f1, 
        "kappa": kappa, 
        "mcc": mcc,
        "precision_per_class": prec_per_class.tolist(),
        "recall_per_class": rec_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist()
    }
    
    if y_prob is not None:
        try:
            y_true_bin = np.zeros((len(y_true), NUM_CLASSES))
            y_true_bin[np.arange(len(y_true)), y_true] = 1
            metrics["roc_auc"] = roc_auc_score(y_true_bin, np.array(y_prob), average="macro")
            # Per-class AUC
            auc_per_class = []
            for i in range(NUM_CLASSES):
                auc_per_class.append(roc_auc_score(y_true_bin[:, i], np.array(y_prob)[:, i]))
            metrics["roc_auc_per_class"] = auc_per_class
        except Exception as e:
            print(f"ROC AUC calculation failed: {e}")
            metrics["roc_auc"] = None
            metrics["roc_auc_per_class"] = None
    return metrics

def plot_confusion(y_true, y_pred, classes, outpath):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize for percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix (Counts)")
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens",
                xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Confusion Matrix (Normalized)")
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_roc_curves(y_true, y_prob, classes, outpath):
    """Plot ROC curves for all classes"""
    y_true_bin = np.zeros((len(y_true), NUM_CLASSES))
    y_true_bin[np.arange(len(y_true)), y_true] = 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    
    for i, color in zip(range(NUM_CLASSES), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{classes[i]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - All Classes')
    ax.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_per_class_metrics(metrics_dict, classes, outpath):
    """Plot per-class precision, recall, F1"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.25
    
    ax.bar(x - width, metrics_dict['precision_per_class'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics_dict['recall_per_class'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics_dict['f1_per_class'], width, label='F1-Score', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ---------------- Training ----------------
history = {
    "train_loss": [], 
    "train_acc": [],
    "val_loss": [], 
    "val_acc": [],
    "val_metrics": [],
    "learning_rates": []
}
best_f1 = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * imgs.size(0)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    val_preds, val_probs, val_targets, val_loss, val_acc = evaluate_model(model, val_loader)
    val_metrics = compute_metrics(val_targets, val_preds, y_prob=val_probs)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_metrics"].append(val_metrics)
    history["learning_rates"].append(optimizer.param_groups[0]['lr'])

    print(f"\nEpoch {epoch}/{NUM_EPOCHS}:")
    print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_metrics['f1']:.4f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    scheduler.step(val_metrics["f1"])

    if val_metrics["f1"] > best_f1:
        best_f1 = val_metrics["f1"]
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        print(f"✅ Saved best model | F1={best_f1:.4f}")

# ---------------- Final Evaluation & Plots ----------------
print("\nGenerating final analysis plots...")

# Load best model for final evaluation
model.load_state_dict(torch.load(SAVE_MODEL_PATH))
final_preds, final_probs, final_targets, _, _ = evaluate_model(model, val_loader)
final_metrics = compute_metrics(final_targets, final_preds, y_prob=final_probs)

# 1. Loss and Accuracy Curves
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Validation Loss")
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy curves
ax2.plot(history["train_acc"], label="Train Acc", linewidth=2)
ax2.plot(history["val_acc"], label="Val Acc", linewidth=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training and Validation Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)

# F1 progression
f1_scores = [m['f1'] for m in history["val_metrics"]]
ax3.plot(f1_scores, label="Val F1", linewidth=2, color='green')
ax3.set_xlabel("Epoch")
ax3.set_ylabel("F1 Score")
ax3.set_title("Validation F1 Score Progression")
ax3.legend()
ax3.grid(alpha=0.3)

# Learning rate
ax4.plot(history["learning_rates"], linewidth=2, color='purple')
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Learning Rate")
ax4.set_title("Learning Rate Schedule")
ax4.set_yscale('log')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=150)
plt.close()

# 2. Final Confusion Matrix
plot_confusion(final_targets, final_preds, class_names, OUT_DIR / "final_confusion_matrix.png")

# 3. ROC Curves
plot_roc_curves(final_targets, final_probs, class_names, OUT_DIR / "roc_curves.png")

# 4. Per-class metrics
plot_per_class_metrics(final_metrics, class_names, OUT_DIR / "per_class_metrics.png")

# 5. Classification Report
report = classification_report(final_targets, final_preds, target_names=class_names, digits=4)
print("\n" + "="*60)
print("FINAL CLASSIFICATION REPORT:")
print("="*60)
print(report)
with open(OUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

# 6. Save all metrics
final_summary = {
    "best_f1": float(best_f1),
    "final_metrics": {k: (float(v) if isinstance(v, (int, float, np.number)) else 
                          [float(x) for x in v] if isinstance(v, (list, np.ndarray)) else v) 
                      for k, v in final_metrics.items()},
    "training_history": {
        "train_loss": [float(x) for x in history["train_loss"]],
        "train_acc": [float(x) for x in history["train_acc"]],
        "val_loss": [float(x) for x in history["val_loss"]],
        "val_acc": [float(x) for x in history["val_acc"]],
        "learning_rates": [float(x) for x in history["learning_rates"]]
    },
    "class_names": class_names,
    "config": {
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "img_size": IMG_SIZE
    }
}

with open(OUT_DIR / "final_metrics.json", "w") as f:
    json.dump(final_summary, f, indent=2)

print(f"\n✅ Training complete. All outputs saved to: {OUT_DIR}")
print(f"📊 Generated plots: training_curves.png, roc_curves.png, per_class_metrics.png, confusion_matrix.png")
print(f"📈 Best F1 Score: {best_f1:.4f}")
