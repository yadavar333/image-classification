#!/usr/bin/env python3
"""
Run inference on a single image using the exported ONNX model.

Usage:
    python inference.py image.jpg
    python inference.py image.jpg --model model.onnx --top 3
"""

import argparse
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

from dataset import CLASSES, _MEAN, _STD


def preprocess(image_path: str) -> np.ndarray:
    """Load and normalise an image to a (1, 3, 32, 32) float32 array."""
    img = Image.open(image_path).convert('RGB').resize((32, 32))
    arr = np.array(img, dtype=np.float32) / 255.0          # HWC, [0,1]

    mean = np.array(_MEAN, dtype=np.float32)
    std  = np.array(_STD,  dtype=np.float32)
    arr  = (arr - mean) / std                               # normalise

    arr  = arr.transpose(2, 0, 1)                           # HWC → CHW
    return arr[np.newaxis, ...]                             # add batch dim


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(image_path: str, model_path: str = 'model.onnx', top_k: int = 3):
    sess   = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp    = preprocess(image_path)
    logits = sess.run(['logits'], {'input': inp})[0][0]     # shape (num_classes,)
    probs  = softmax(logits)

    top_indices = probs.argsort()[::-1][:top_k]

    print(f'\nImage  : {image_path}')
    print(f'Model  : {model_path}')
    print(f'\nTop-{top_k} predictions:')
    for rank, idx in enumerate(top_indices, 1):
        print(f'  {rank}. {CLASSES[idx]:12s}  {probs[idx]*100:6.2f}%')

    return CLASSES[top_indices[0]], float(probs[top_indices[0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify an image with the CIFAR-10 5-class CNN')
    parser.add_argument('image',          help='Path to input image')
    parser.add_argument('--model', '-m',  default='model.onnx', help='ONNX model file')
    parser.add_argument('--top',   '-k',  type=int, default=3,  help='Show top-k predictions')
    args = parser.parse_args()

    if not args.image:
        parser.print_help()
        sys.exit(1)

    predict(args.image, args.model, args.top)
