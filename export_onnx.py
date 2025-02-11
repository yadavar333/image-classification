"""Export a trained PyTorch checkpoint to ONNX format."""

import argparse

import torch

from dataset import NUM_CLASSES
from model import CNN


def export(checkpoint: str = 'best_model.pt', output: str = 'model.onnx'):
    device = torch.device('cpu')

    model = CNN(num_classes=NUM_CLASSES)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)

    torch.onnx.export(
        model,
        dummy_input,
        output,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input':  {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        },
        opset_version=17,
    )

    print(f'Model exported → {output}')
    print(f'  Input  : (batch, 3, 32, 32)')
    print(f'  Output : (batch, {NUM_CLASSES})  — raw logits, apply softmax for probabilities')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export CNN checkpoint to ONNX')
    parser.add_argument('--checkpoint', default='best_model.pt')
    parser.add_argument('--output',     default='model.onnx')
    args = parser.parse_args()
    export(args.checkpoint, args.output)
