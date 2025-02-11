# Image Classification Pipeline

Custom CNN trained on a 5-class subset of CIFAR-10. Full pipeline: data augmentation → training with W&B logging → confusion matrix evaluation → ONNX export → inference CLI.

## Stack
PyTorch · torchvision · ONNX · onnxruntime · Weights & Biases · scikit-learn · seaborn

## Classes
`airplane` · `automobile` · `bird` · `cat` · `deer`  (CIFAR-10 indices 0–4)

## Architecture

```
Input  (3 × 32 × 32)
   │
   ▼
ConvBlock 1 ──  Conv2d(3→32, k=3, pad=1) → BatchNorm → ReLU → MaxPool(2)
   │                                                           32 × 16 × 16
   ▼
ConvBlock 2 ──  Conv2d(32→64, k=3, pad=1) → BatchNorm → ReLU → MaxPool(2)
   │                                                            64 × 8 × 8
   ▼
ConvBlock 3 ──  Conv2d(64→128, k=3, pad=1) → BatchNorm → ReLU → MaxPool(2)
   │                                                            128 × 4 × 4
   ▼
Flatten  →  Linear(2048→256)  →  ReLU  →  Dropout(0.5)  →  Linear(256→5)
   │
   ▼
Logits (5)
```

Trainable parameters: ~568 k

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1 — Train

```bash
python train.py                          # defaults: 30 epochs, lr=1e-3, patience=5
python train.py --epochs 50 --wandb      # with W&B logging
```

Training downloads CIFAR-10 automatically into `./data/`.  
Best checkpoint saved to `best_model.pt`.

### 2 — Evaluate

```bash
python evaluate.py --checkpoint best_model.pt
```

Outputs to `./outputs/`:
- `confusion_matrix.png`
- `misclassified.png`
- `classification_report.txt`

### 3 — Export to ONNX

```bash
python export_onnx.py --checkpoint best_model.pt --output model.onnx
```

### 4 — Inference

```bash
python inference.py image.jpg
python inference.py image.jpg --model model.onnx --top 3
```

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (weight_decay=1e-4) |
| Initial LR | 1e-3 |
| Scheduler | StepLR (step=10, γ=0.1) |
| Early stopping | patience=5 |
| Batch size | 64 |
| Augmentation | RandomHorizontalFlip, RandomCrop(32,pad=4), ColorJitter |

## Expected Results

| Metric | Value |
|--------|-------|
| Val accuracy | ~88–91% |
| Val loss (best) | ~0.30–0.35 |
| Training time (CPU) | ~15 min / 30 epochs |
| Training time (GPU) | ~3 min / 30 epochs |

## W&B Integration

Pass `--wandb` to `train.py` to log metrics to Weights & Biases.  
Set `WANDB_API_KEY` environment variable or run `wandb login` first.  
W&B is optional — training runs normally without it.
