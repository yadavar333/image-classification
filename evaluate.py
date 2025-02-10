"""Evaluation: confusion matrix heatmap, per-class report, misclassified samples grid."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from dataset import get_dataloaders, CLASSES, NUM_CLASSES, _MEAN, _STD
from model import CNN


def load_model(checkpoint: str, device: torch.device) -> CNN:
    model = CNN(num_classes=NUM_CLASSES).to(device)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded checkpoint: {checkpoint}  (val_loss={ckpt["val_loss"]:.4f}, val_acc={ckpt["val_acc"]:.4f})')
    return model


def collect_predictions(model, val_loader, device):
    all_preds, all_labels, all_images = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            all_images.append(images.cpu())
    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_images),
    )


def plot_confusion_matrix(y_true, y_pred, out_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_misclassified(images, y_true, y_pred, out_dir: str, n: int = 16):
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print('No misclassified samples found.')
        return

    wrong_idx = wrong_idx[:n]
    cols = 4
    rows = (len(wrong_idx) + cols - 1) // cols

    # Denormalise for display
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std  = torch.tensor(_STD).view(3, 1, 1)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(-1)

    for i, idx in enumerate(wrong_idx):
        img = images[idx] * std + mean           # denormalise
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[i].imshow(img)
        axes[i].set_title(
            f'True: {CLASSES[y_true[idx]]}\nPred: {CLASSES[y_pred[idx]]}',
            fontsize=7, color='red',
        )
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Misclassified Samples', fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, 'misclassified.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def evaluate(checkpoint: str = 'best_model.pt',
             data_dir: str   = './data',
             out_dir: str    = './outputs'):

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(checkpoint, device)
    _, val_loader = get_dataloaders(data_dir=data_dir, batch_size=128)

    y_pred, y_true, images = collect_predictions(model, val_loader, device)

    # ── Per-class classification report ───────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print('\nClassification Report:\n')
    print(report)
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # ── Confusion matrix ──────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, out_dir)

    # ── Misclassified grid ────────────────────────────────────────────────
    plot_misclassified(images, y_true, y_pred, out_dir)

    overall_acc = (y_pred == y_true).mean()
    print(f'\nOverall accuracy: {overall_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained CNN checkpoint')
    parser.add_argument('--checkpoint', default='best_model.pt')
    parser.add_argument('--data-dir',   default='./data')
    parser.add_argument('--out-dir',    default='./outputs')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.out_dir)
