"""Training loop with W&B logging, StepLR scheduler, and early stopping."""

import argparse
import os

import torch
import torch.nn as nn
from torch import optim

from dataset import get_dataloaders, NUM_CLASSES
from model import CNN, count_params

# Optional W&B — skipped gracefully if not installed / not logged in
try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


def train(
    epochs:      int   = 30,
    batch_size:  int   = 64,
    lr:          float = 1e-3,
    patience:    int   = 5,
    checkpoint:  str   = 'best_model.pt',
    data_dir:    str   = './data',
    use_wandb:   bool  = False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── W&B init ──────────────────────────────────────────────────────────
    if use_wandb and _WANDB:
        wandb.init(
            project='cifar10-5class',
            config=dict(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience),
        )

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────
    model = CNN(num_classes=NUM_CLASSES).to(device)
    print(f'Trainable params: {count_params(model):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float('inf')
    patience_ctr  = 0

    for epoch in range(1, epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = train_correct = train_total = 0

        for images, labels in train_loader:
            # Remap CIFAR-10 labels (0-4 are already correct for our subset)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = val_correct = val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        train_loss /= train_total
        val_loss   /= val_total
        train_acc   = train_correct / train_total
        val_acc     = val_correct   / val_total

        scheduler.step()

        print(
            f'Epoch {epoch:3d}/{epochs}  '
            f'train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  '
            f'val_loss={val_loss:.4f}  val_acc={val_acc:.4f}'
        )

        if use_wandb and _WANDB:
            wandb.log(dict(
                epoch=epoch,
                train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss,     val_acc=val_acc,
                lr=scheduler.get_last_lr()[0],
            ))

        # ── Checkpoint + early stopping ───────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save({
                'epoch':      epoch,
                'model_state': model.state_dict(),
                'val_loss':   val_loss,
                'val_acc':    val_acc,
            }, checkpoint)
            print(f'  ✓ checkpoint saved (val_loss={val_loss:.4f})')
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f'Early stopping at epoch {epoch} (patience={patience})')
                break

    if use_wandb and _WANDB:
        wandb.finish()

    print(f'\nBest val_loss: {best_val_loss:.4f}  — checkpoint: {checkpoint}')
    return checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN on CIFAR-10 5-class subset')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch-size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--patience',   type=int,   default=5)
    parser.add_argument('--checkpoint', type=str,   default='best_model.pt')
    parser.add_argument('--data-dir',   type=str,   default='./data')
    parser.add_argument('--wandb',      action='store_true', help='Enable W&B logging')
    args = parser.parse_args()

    train(
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
        checkpoint=args.checkpoint, data_dir=args.data_dir,
        use_wandb=args.wandb,
    )
