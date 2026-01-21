# ============================================================
# ResNet50 - Testing Script for 6 Class Real-vs-Fake
# ============================================================
# Load trained model and evaluate on test set
# Generates comprehensive analysis and visualizations
# ============================================================

import os
import json
from pathlib import Path
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
import random

# ---------------- USER CONFIG ----------------
MODEL_PATH = Path("/kaggle/input/your-model-dataset/resnet50_6class.pth")  # UPDATE THIS
OUT_DIR = Path("/kaggle/working/resnet50_test_output")
NUM_CLASSES = 6
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_WORKERS = 4
USE_AMP = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Test dataset paths ---
test_dirs = {
    "real": r"/kaggle/input/real-vs-fake/model/test/test real",
    "deepfake": r"/kaggle/input/real-vs-fake/model/test/test deepfake",
    "stylegan": r"/kaggle/input/real-vs-fake/model/test/test style gan",
    "stylegan2": r"/kaggle/input/real-vs-fake/model/test/test style gan 2",
    "pggan": r"/kaggle/input/real-vs-fake/model/test/test pggan",
    "stablediffusion": r"/kaggle/input/real-vs-fake/model/test/test stable diffusion",
}
# ------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Dataset ----------------
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
        return img, label, img_path  # Return path for error analysis

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
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- Load Test Data ----------------
test_data, test_labels, class_names, test_counts = load_custom_split(test_dirs, test_transforms)
test_dataset = CustomImageDataset(test_data, test_labels, test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

print(f"Loaded {len(test_dataset)} test images.")
print("Classes:", class_names)
print("Test Distribution:", test_counts)

# Plot test distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(test_counts.keys(), test_counts.values())
ax.set_title("Test Set Distribution")
ax.set_ylabel("Number of Images")
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "test_distribution.png", dpi=150)
plt.close()

# ---------------- Load Model ----------------
print(f"\nLoading model from: {MODEL_PATH}")
model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully")

# ---------------- Evaluation ----------------
def evaluate_model(model, loader):
    """Full evaluation with per-sample tracking"""
    model.eval()
    preds, probs, targets, paths = [], [], [], []
    
    with torch.no_grad():
        for imgs, labels, img_paths in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(imgs)
            
            soft = F.softmax(out, dim=1).cpu().numpy()
            pred_labels = np.argmax(soft, axis=1)
            
            preds.extend(pred_labels.tolist())
            probs.extend(soft.tolist())
            targets.extend(labels.cpu().numpy().tolist())
            paths.extend(img_paths)
    
    return np.array(preds), np.array(probs), np.array(targets), paths

print("\nRunning inference on test set...")
test_preds, test_probs, test_targets, test_paths = evaluate_model(model, test_loader)

# ---------------- Compute Metrics ----------------
def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
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
            auc_per_class = []
            for i in range(NUM_CLASSES):
                auc_per_class.append(roc_auc_score(y_true_bin[:, i], np.array(y_prob)[:, i]))
            metrics["roc_auc_per_class"] = auc_per_class
        except Exception as e:
            print(f"ROC AUC calculation failed: {e}")
            metrics["roc_auc"] = None
            metrics["roc_auc_per_class"] = None
    return metrics

test_metrics = compute_metrics(test_targets, test_preds, y_prob=test_probs)

print("\n" + "="*60)
print("TEST SET RESULTS:")
print("="*60)
print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall:    {test_metrics['recall']:.4f}")
print(f"F1 Score:  {test_metrics['f1']:.4f}")
print(f"Kappa:     {test_metrics['kappa']:.4f}")
print(f"MCC:       {test_metrics['mcc']:.4f}")
if test_metrics['roc_auc']:
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")

# ---------------- Visualization Functions ----------------
def plot_confusion(y_true, y_pred, classes, outpath):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Test Confusion Matrix (Counts)")
    
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens",
                xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Test Confusion Matrix (Normalized)")
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_roc_curves(y_true, y_prob, classes, outpath):
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
    ax.set_title('Test Set ROC Curves - All Classes')
    ax.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_per_class_metrics(metrics_dict, classes, outpath):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.25
    
    ax.bar(x - width, metrics_dict['precision_per_class'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics_dict['recall_per_class'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics_dict['f1_per_class'], width, label='F1-Score', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Test Set - Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_confidence_distribution(y_true, y_prob, classes, outpath):
    """Plot confidence distribution for correct vs incorrect predictions"""
    y_pred = np.argmax(y_prob, axis=1)
    max_probs = np.max(y_prob, axis=1)
    
    correct_mask = (y_pred == y_true)
    correct_conf = max_probs[correct_mask]
    incorrect_conf = max_probs[~correct_mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall confidence distribution
    ax1.hist(correct_conf, bins=30, alpha=0.6, label='Correct', color='green')
    ax1.hist(incorrect_conf, bins=30, alpha=0.6, label='Incorrect', color='red')
    ax1.set_xlabel('Confidence (Max Probability)')
    ax1.set_ylabel('Count')
    ax1.set_title('Prediction Confidence Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Average confidence per class
    avg_conf_per_class = []
    for i in range(len(classes)):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            avg_conf_per_class.append(max_probs[class_mask].mean())
        else:
            avg_conf_per_class.append(0)
    
    ax2.bar(classes, avg_conf_per_class, alpha=0.7)
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('Average Prediction Confidence per Class')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def find_worst_predictions(y_true, y_pred, y_prob, paths, classes, top_n=20):
    """Find most confidently wrong predictions"""
    max_probs = np.max(y_prob, axis=1)
    incorrect_mask = (y_pred != y_true)
    
    incorrect_indices = np.where(incorrect_mask)[0]
    incorrect_confidences = max_probs[incorrect_mask]
    
    # Sort by confidence (descending)
    sorted_indices = incorrect_indices[np.argsort(-incorrect_confidences)]
    
    worst_cases = []
    for idx in sorted_indices[:top_n]:
        worst_cases.append({
            "path": paths[idx],
            "true_label": classes[y_true[idx]],
            "pred_label": classes[y_pred[idx]],
            "confidence": float(max_probs[idx]),
            "true_prob": float(y_prob[idx, y_true[idx]]),
            "pred_prob": float(y_prob[idx, y_pred[idx]])
        })
    
    return worst_cases

def visualize_error_samples(worst_cases, outpath, num_samples=12):
    """Visualize most confident errors"""
    n_rows = 3
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(worst_cases))):
        case = worst_cases[i]
        img = Image.open(case['path']).convert('RGB')
        
        axes[i].imshow(img)
        axes[i].axis('off')
        title = f"True: {case['true_label']}\n"
        title += f"Pred: {case['pred_label']} ({case['confidence']:.2%})"
        axes[i].set_title(title, fontsize=9, color='red')
    
    # Hide unused subplots
    for i in range(len(worst_cases), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Most Confident Misclassifications", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ---------------- Generate All Plots ----------------
print("\nGenerating visualizations...")

# 1. Confusion matrix
plot_confusion(test_targets, test_preds, class_names, OUT_DIR / "test_confusion_matrix.png")

# 2. ROC curves
plot_roc_curves(test_targets, test_probs, class_names, OUT_DIR / "test_roc_curves.png")

# 3. Per-class metrics
plot_per_class_metrics(test_metrics, class_names, OUT_DIR / "test_per_class_metrics.png")

# 4. Confidence distribution
plot_confidence_distribution(test_targets, test_probs, class_names, OUT_DIR / "confidence_distribution.png")

# 5. Classification report
report = classification_report(test_targets, test_preds, target_names=class_names, digits=4)
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT:")
print("="*60)
print(report)
with open(OUT_DIR / "test_classification_report.txt", "w") as f:
    f.write(report)

# 6. Error analysis
print("\nPerforming error analysis...")
worst_predictions = find_worst_predictions(test_targets, test_preds, test_probs, 
                                          test_paths, class_names, top_n=50)

# Save worst predictions to JSON
with open(OUT_DIR / "worst_predictions.json", "w") as f:
    json.dump(worst_predictions[:50], f, indent=2)

# Visualize top errors
visualize_error_samples(worst_predictions, OUT_DIR / "worst_predictions_visualization.png", num_samples=12)

# 7. Per-class error analysis
print("\nPer-class error analysis:")
cm = confusion_matrix(test_targets, test_preds)
for i, class_name in enumerate(class_names):
    total = cm[i].sum()
    correct = cm[i, i]
    accuracy = correct / total if total > 0 else 0
    print(f"  {class_name:15s}: {correct:4d}/{total:4d} correct ({accuracy:.2%})")
    
    # Most confused with
    if total > 0:
        confused_idx = np.argsort(cm[i])[::-1]
        confused_idx = [idx for idx in confused_idx if idx != i][:2]
        for confused_class in confused_idx:
            if cm[i, confused_class] > 0:
                print(f"    → Confused with {class_names[confused_class]}: {cm[i, confused_class]} times")

# 8. Save all metrics
test_summary = {
    "test_metrics": {k: (float(v) if isinstance(v, (int, float, np.number)) else 
                        [float(x) for x in v] if isinstance(v, (list, np.ndarray)) else v) 
                    for k, v in test_metrics.items()},
    "class_names": class_names,
    "test_counts": test_counts,
    "total_samples": len(test_dataset),
    "model_path": str(MODEL_PATH),
    "confusion_matrix": cm.tolist()
}

with open(OUT_DIR / "test_results.json", "w") as f:
    json.dump(test_summary, f, indent=2)

# Archive everything
import shutil
print("\nArchiving results...")
shutil.make_archive('/kaggle/working/resnet50_test_results', 'zip', OUT_DIR)

print(f"\n✅ Testing complete. All outputs saved to: {OUT_DIR}")
print(f"📦 Results archived to: /kaggle/working/resnet50_test_results.zip")
print(f"📊 Generated plots:")
print(f"   - test_confusion_matrix.png")
print(f"   - test_roc_curves.png")
print(f"   - test_per_class_metrics.png")
print(f"   - confidence_distribution.png")
print(f"   - worst_predictions_visualization.png")
print(f"📈 Test F1 Score: {test_metrics['f1']:.4f}")
print(f"📉 Test Accuracy: {test_metrics['accuracy']:.4f}")
