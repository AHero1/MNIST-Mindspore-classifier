# MNIST-Mindspore-classifier
# MNIST Classifier with MindSpore

This project demonstrates a simple digit classifier trained on the MNIST dataset using [MindSpore](https://www.mindspore.cn/), a deep learning framework.

## Model

- Input: 28x28 grayscale images
- Architecture: Fully connected neural network with ReLU
- Output: 10-class softmax for digits 0–9
- Accuracy: ~97.7% on test set

## Requirements

- mindspore
- numpy

Install dependencies:

```bash
pip install mindspore numpy
