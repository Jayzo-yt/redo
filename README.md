# ResNet50 Testing Script for Real-vs-Fake Classification

This repository contains a Python script for testing a pre-trained ResNet50 model on a 6-class real-vs-fake image classification task. It evaluates the model on a local test dataset, generates comprehensive metrics, and produces visualizations for analysis.

## Features
- Loads a trained ResNet50 model and evaluates it on custom test data.
- Supports 6 classes: real, deepfake, stylegan, stylegan2, pggan, stablediffusion.
- Computes accuracy, precision, recall, F1-score, Cohen's Kappa, MCC, and ROC-AUC.
- Generates confusion matrices, ROC curves, per-class metrics, confidence distributions, and error analysis.
- Saves results to a specified output directory.

## Prerequisites
- Python 3.8–3.11 (64-bit recommended).
- PyTorch (CPU or GPU).
- Other dependencies: torchvision, PIL, scikit-learn, tqdm, matplotlib, seaborn, numpy.

## Quick Start

### 1. Install Requirements
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. For GPU Support (NVIDIA CUDA)
If you have an NVIDIA GPU with CUDA installed, install PyTorch with CUDA support (adjust the URL based on your CUDA version, e.g., cu118 for CUDA 11.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Prepare Your Data
- Update the paths in `RESNET_TEST_LOCAL.py`:
  - `MODEL_PATH`: Path to your trained ResNet50 model (`.pth` file).
  - `TEST_DATA_ROOT`: Root directory for test data.
  - `test_dirs`: Dictionary of class names to their respective folder paths (supports subfolders like "high" and "low").
  - `OUT_DIR`: Directory to save results.

Ensure your test folders contain images in formats like `.jpg`, `.jpeg`, or `.png`.

### 4. Run the Script
Execute the testing script:

```bash
python RESNET_TEST_LOCAL.py
```

The script will validate paths, load data, run inference, compute metrics, and generate visualizations. Results are saved to `OUT_DIR`.

## Output Files
- `test_distribution.png`: Bar chart of test set distribution.
- `test_confusion_matrix.png`: Confusion matrix (counts and normalized).
- `test_roc_curves.png`: ROC curves for all classes.
- `test_per_class_metrics.png`: Per-class precision, recall, and F1-score.
- `confidence_distribution.png`: Prediction confidence analysis.
- `worst_predictions_visualization.png`: Visualization of top misclassifications.
- `test_classification_report.txt`: Detailed classification report.
- `worst_predictions.json`: JSON of worst predictions.
- `test_results.json`: Summary of all metrics and results.

## Troubleshooting
- **No test data loaded**: Verify paths in `test_dirs` and ensure images are in subfolders.
- **Torch import error**: Install Visual C++ Redistributables and reinstall PyTorch.
- **CUDA issues**: Ensure compatible CUDA version and drivers.

## License
[Add your license here, e.g., MIT]

## Contributing
[Add contribution guidelines if applicable]
