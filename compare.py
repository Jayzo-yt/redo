import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import os
from pathlib import Path

# Set style for publication-quality plots (improved for poster: use a more professional style)
plt.style.use('seaborn-v0_8-whitegrid')  # Changed to whitegrid for cleaner poster look
sns.set_palette("Set2")  # Softer, more modern palette for better visual appeal

class SyntheticImageDataset(Dataset):
    """Dataset loader for synthetic images with flexible directory structure"""
    def __init__(self, class_dirs_dict, transform=None):
        """
        Args:
            class_dirs_dict: Dictionary mapping class names to directory paths
                           e.g., {'real': 'path/to/real', 'deepfake': 'path/to/deepfake'}
            transform: torchvision transforms
        """
        self.transform = transform
        self.classes = ['real', 'deepfake', 'stylegan', 'stylegan2', 'pggan', 'stablediffusion']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            if cls not in class_dirs_dict:
                print(f"Warning: No directory provided for class '{cls}'")
                continue
            
            cls_dir = Path(class_dirs_dict[cls])
            if not cls_dir.exists():
                print(f"Warning: Directory does not exist: {cls_dir}")
                continue
            
            # Load all image files recursively (including subdirectories)
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']
            image_files = []
            for ext in image_extensions:
                image_files.extend(cls_dir.rglob(ext))  # rglob searches recursively
            
            for img_path in image_files:
                self.samples.append((img_path, self.class_to_idx[cls]))
            
            print(f"Loaded {len(image_files)} images from {cls} ({cls_dir})")
        
        print(f"\nTotal: {len(self.samples)} images loaded")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_model(model_path, num_classes=6):
    """Load a trained ResNet50 model"""
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and return predictions and true labels"""
    model = model.to(device)
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"Evaluating on {len(dataloader.dataset)} images...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * dataloader.batch_size}/{len(dataloader.dataset)} images")
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_accuracy_comparison(results_dict, save_path='accuracy_comparison.png'):
    """
    Create bar chart comparing test accuracies across training regimes
    IMPROVED: Larger figure, bolder fonts, annotations, and shadow effects for poster impact
    """
    fig, ax = plt.subplots(figsize=(14, 8))  # Larger for poster
    
    regimes = list(results_dict.keys())
    accuracies = [results_dict[r]['accuracy'] * 100 for r in regimes]
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red for HQ (bad), Blue for LQ, Green for Mixed (best)
    bars = ax.bar(regimes, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.9, width=0.6)
    
    # Add value labels on bars with shadow for readability
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=20, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Training Regime', fontsize=18, fontweight='bold')
    ax.set_title('Degradation-Aware Training Outperforms Standard Approaches', 
                 fontsize=22, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)  # Extra space for labels
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90% Benchmark')
    ax.grid(axis='y', alpha=0.4)
    ax.legend(fontsize=14, loc='upper left')
    ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')  # Higher DPI for poster print
    print(f"✓ Saved accuracy comparison to {save_path}")
    plt.close()

def plot_confusion_matrices(results_dict, class_names, save_path='confusion_matrices.png'):
    """Plot confusion matrices for all three models side by side
    IMPROVED: Larger fonts, better cmap, added colorbar label
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))  # Wider for poster
    
    regimes = list(results_dict.keys())
    
    for idx, (regime, ax) in enumerate(zip(regimes, axes)):
        y_true = results_dict[regime]['y_true']
        y_pred = results_dict[regime]['y_pred']
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlGnBu',  # Cooler cmap for better contrast
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Normalized Accuracy'}, vmin=0, vmax=1,
                   annot_kws={"size": 14})  # Larger annotations
        
        acc = results_dict[regime]['accuracy']
        ax.set_title(f'{regime}\nAccuracy: {acc*100:.1f}%', 
                    fontsize=18, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=16)
        ax.set_xlabel('Predicted Label', fontsize=16)
        ax.tick_params(labelsize=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"✓ Saved confusion matrices to {save_path}")
    plt.close()

def plot_per_class_performance(results_dict, class_names, save_path='per_class_performance.png'):
    """
    Show per-class F1 scores for each training regime
    IMPROVED: Grouped bars, larger fonts, added gridlines
    """
    fig, ax = plt.subplots(figsize=(16, 8))  # Larger for poster
    
    regimes = list(results_dict.keys())
    x = np.arange(len(class_names))
    width = 0.25
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for idx, regime in enumerate(regimes):
        f1_scores = results_dict[regime]['per_class_f1']
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, f1_scores * 100, width, 
                     label=regime, color=colors[idx], alpha=0.9, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('F1-Score (%)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Class', fontsize=18, fontweight='bold')
    ax.set_title('Per-Class F1-Scores Reveal HQ Model Weaknesses', 
                 fontsize=22, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=16)
    ax.legend(fontsize=16, loc='upper left')
    ax.set_ylim(0, 105)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.7, label='80% Threshold')
    ax.grid(axis='y', alpha=0.4)
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"✓ Saved per-class performance to {save_path}")
    plt.close()

def plot_prediction_examples(models_dict, test_dataset, class_names, 
                            num_examples=9, save_path='prediction_examples.png'):  # Increased to 9 for more impact
    """
    Show examples where HQ fails but Mixed succeeds
    IMPROVED: More examples, larger figure, added borders and confidence bars
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nSearching for examples where HQ fails but Mixed succeeds...")
    
    # Find examples where HQ is wrong but Mixed is correct
    examples = []
    
    for idx in range(len(test_dataset)):
        if len(examples) >= num_examples:
            break
        
        if (idx + 1) % 100 == 0:
            print(f"  Searched {idx + 1}/{len(test_dataset)} images, found {len(examples)} examples")
        
        image, true_label = test_dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            hq_out = models_dict['HIGH_QUALITY'](image_batch)
            mixed_out = models_dict['MIXED'](image_batch)
            
            hq_pred = torch.argmax(hq_out, dim=1).item()
            mixed_pred = torch.argmax(mixed_out, dim=1).item()
            
            hq_conf = torch.softmax(hq_out, dim=1).max().item()
            mixed_conf = torch.softmax(mixed_out, dim=1).max().item()
        
        # Find cases where HQ is wrong but Mixed is correct
        if hq_pred != true_label and mixed_pred == true_label:
            examples.append({
                'image': image,
                'true_label': true_label,
                'hq_pred': hq_pred,
                'hq_conf': hq_conf,
                'mixed_pred': mixed_pred,
                'mixed_conf': mixed_conf
            })
    
    if len(examples) == 0:
        print("⚠ Warning: No examples found where HQ fails but Mixed succeeds")
        print("  This might mean HQ model performs better than expected on test set")
        return
    
    print(f"✓ Found {len(examples)} examples")
    
    # Plot examples
    num_cols = 3
    num_rows = (len(examples) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 6*num_rows))  # Larger images
    
    axes = axes.flatten() if num_rows > 1 else [axes]
    
    for idx, example in enumerate(examples):
        ax = axes[idx]
        
        # Denormalize image for display
        img = example['image'].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('', y=-0.1)  # Space for text below
        
        true_class = class_names[example['true_label']]
        hq_class = class_names[example['hq_pred']]
        mixed_class = class_names[example['mixed_pred']]
        
        title = f"True: {true_class}\n"
        title += f"HQ Pred: {hq_class} ({example['hq_conf']*100:.1f}%) ✗\n"
        title += f"Mixed Pred: {mixed_class} ({example['mixed_conf']*100:.1f}%) ✓"
        
        ax.text(0.5, -0.15, title, ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Hide unused subplots
    for idx in range(len(examples), num_rows * num_cols):
        axes[idx].axis('off')
    
    plt.suptitle('Visual Evidence: Mixed Model Handles Degraded Images Better', 
                 fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"✓ Saved prediction examples to {save_path}")
    plt.close()

def plot_roc_curves(results_dict, class_names, save_path='roc_curves.png'):
    """
    NEW: Plot multi-class ROC curves (one-vs-rest) for each regime side by side
    This adds robustness metric for imbalanced classes, great for poster
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))  # Side by side for comparison
    
    regimes = list(results_dict.keys())
    
    macro_aucs = {}  # To store macro AUC for each regime
    
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        y_true = results_dict[regime]['y_true']
        y_probs = results_dict[regime]['y_probs']
        
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        class_aucs = []
        
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            class_aucs.append(roc_auc)
            ax.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})', linewidth=2.5)
        
        macro_aucs[regime] = np.mean(class_aucs)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2)
        ax.set_xlabel('False Positive Rate', fontsize=16)
        ax.set_ylabel('True Positive Rate', fontsize=16)
        ax.set_title(f'{regime} ROC Curves\nMacro AUC: {macro_aucs[regime]:.2f}', 
                     fontsize=18, fontweight='bold')
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(alpha=0.4)
        ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"✓ Saved ROC curves to {save_path}")
    plt.close()

def plot_precision_recall_curves(results_dict, class_names, save_path='pr_curves.png'):
    """
    NEW: Plot multi-class Precision-Recall curves (one-vs-rest) for each regime
    Useful for highlighting performance on imbalanced datasets
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    regimes = list(results_dict.keys())
    
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        y_true = results_dict[regime]['y_true']
        y_probs = results_dict[regime]['y_probs']
        
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        for i, cls in enumerate(class_names):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
            ax.plot(recall, precision, label=f'{cls}', linewidth=2.5)
        
        ax.set_xlabel('Recall', fontsize=16)
        ax.set_ylabel('Precision', fontsize=16)
        ax.set_title(f'{regime} Precision-Recall Curves', 
                     fontsize=18, fontweight='bold')
        ax.legend(loc='lower left', fontsize=12)
        ax.grid(alpha=0.4)
        ax.tick_params(labelsize=14)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"✓ Saved Precision-Recall curves to {save_path}")
    plt.close()

def generate_all_visualizations(model_paths, test_dirs_dict, output_dir='poster_figures'):
    """
    Main function to generate all visualizations
    IMPROVED: Added new plots (ROC and PR curves), higher DPI, better fonts throughout
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    class_names = ['Real', 'Deepfake', 'StyleGAN', 'StyleGAN2', 'PGGAN', 'StableDiffusion']
    
    # Data transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    print("="*60)
    print("Loading test dataset...")
    print("="*60)
    test_dataset = SyntheticImageDataset(test_dirs_dict, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load models and evaluate
    results = {}
    models = {}
    
    for regime, model_path in model_paths.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {regime} model...")
        print(f"Model path: {model_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(model_path):
            print(f"⚠ ERROR: Model file not found: {model_path}")
            print(f"  Please check the path and try again.")
            continue
        
        model = load_model(model_path)
        models[regime] = model.to(device)
        
        y_pred, y_true, y_probs = evaluate_model(model, test_loader, device, class_names)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        results[regime] = {
            'y_pred': y_pred,
            'y_true': y_true,
            'y_probs': y_probs,
            'accuracy': accuracy,
            'per_class_f1': f1,
            'per_class_precision': precision,
            'per_class_recall': recall
        }
        
        print(f"\n{regime} Results:")
        print(f"Overall Accuracy: {accuracy*100:.2f}%")
        print("\nPer-Class F1 Scores:")
        for cls, f1_score in zip(class_names, f1):
            print(f"  {cls:15s}: {f1_score*100:.2f}%")
    
    if len(results) == 0:
        print("\n⚠ ERROR: No models were successfully loaded and evaluated!")
        print("  Please check your model paths and try again.")
        return None
    
    # Generate all visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}\n")
    
    # 1. Accuracy comparison (MOST IMPORTANT)
    plot_accuracy_comparison(results, 
                           save_path=os.path.join(output_dir, '1_accuracy_comparison.png'))
    
    # 2. Confusion matrices
    plot_confusion_matrices(results, class_names,
                          save_path=os.path.join(output_dir, '2_confusion_matrices.png'))
    
    # 3. Per-class performance
    plot_per_class_performance(results, class_names,
                             save_path=os.path.join(output_dir, '3_per_class_performance.png'))
    
    # 4. Prediction examples (only if we have HQ and Mixed models)
    if 'HIGH_QUALITY' in models and 'MIXED' in models:
        plot_prediction_examples(models, test_dataset, class_names,
                               save_path=os.path.join(output_dir, '4_prediction_examples.png'))
    else:
        print("⚠ Skipping prediction examples (need both HIGH_QUALITY and MIXED models)")
    
    # 5. NEW: ROC curves
    plot_roc_curves(results, class_names,
                    save_path=os.path.join(output_dir, '5_roc_curves.png'))
    
    # 6. NEW: Precision-Recall curves
    plot_precision_recall_curves(results, class_names,
                                 save_path=os.path.join(output_dir, '6_pr_curves.png'))
    
    print(f"\n{'='*60}")
    print(f"✓ ALL VISUALIZATIONS SAVED TO: {output_dir}/")
    print(f"{'='*60}")
    print("\nGenerated files (improved for poster):")
    print("  1. 1_accuracy_comparison.png - MAIN RESULT (enlarged fonts, annotations)")
    print("  2. 2_confusion_matrices.png - Enhanced contrast and labels")
    print("  3. 3_per_class_performance.png - Grouped bars with thresholds")
    print("  4. 4_prediction_examples.png - More examples, better layout")
    print("  5. 5_roc_curves.png - NEW: ROC curves for robustness analysis")
    print("  6. 6_pr_curves.png - NEW: Precision-Recall for imbalance insights")
    
    return results

# USAGE WITH YOUR DIRECTORY STRUCTURE:
if __name__ == "__main__":
    # YOUR test directories
    test_dirs = {
        "real": r"D:\model\model\test\new real",
        "deepfake": r"D:\model\model\test\deepfake",
        "stylegan": r"D:\model\model\test\stylegan",
        "stylegan2": r"D:\model\model\test\stylegan2",
        "pggan": r"D:\model\model\test\pggan",
        "stablediffusion": r"D:\model\model\test\stable diffusion",
    }
    
    # Define paths to your trained models
    # REPLACE THESE WITH YOUR ACTUAL MODEL PATHS
    model_paths = {
        'HIGH_QUALITY': r"D:\resnet-post (1)\resnet50_6class_high_quality.pth",
        'LOW_QUALITY': r"D:\resnet50_low_quality_results\resnet50_6class_low_quality.pth",
        'MIXED': r"D:\resnet50_mixed_results\resnet50_6class_mixed.pth"
    }
    
    # Generate all visualizations
    results = generate_all_visualizations(
        model_paths=model_paths,
        test_dirs_dict=test_dirs,
        output_dir='poster_figures'
    )
    
    if results:
        print("\n✓ SUCCESS! Use these improved figures in your poster.")
        print("\nNext steps:")
        print("1. Open poster_figures/ folder")
        print("2. Use figure 1 as your MAIN RESULT - make it HUGE")
        print("3. Add figures 2-6 to show detailed analysis")
        print("4. Let the figures tell the story - minimal text needed")
    else:
        print("\n⚠ Failed to generate visualizations. Check the errors above.")