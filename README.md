# Handwriting Recognition using PyTorch

A comprehensive deep learning project showcasing the design, implementation, and
deployment of a convolutional neural network for digit classification.

> Full write-up: https://www.milanghimire.info.np/projects/handwriting-recognition-pytorch

## Overview

This project builds a CNN in PyTorch that recognizes handwritten digits (0–9)
directly from raw image pixels without manual feature extraction. The model trains
on the MNIST dataset and achieves **93.46% accuracy** on 10,000 unseen test images.
Training metrics are monitored in real-time using TensorBoard.

## Core Concepts

**Deep Learning**: A machine learning approach that stacks multiple layers of
mathematical units (neurons) to automatically learn complex patterns from data by
adjusting internal weights.

**Neural Networks**: Functions with tunable parameters that transform input images
into output scores—one per digit class. Training adjusts these parameters until
predictions match correct answers.

**Convolutional Neural Networks (CNNs)**: Networks that respect 2D image structure
by sliding small filters (kernels) across images to detect patterns hierarchically:
- Layer 1: Edges and strokes
- Layer 2: Corners, loops, junctions
- Fully-connected layers: Whole-digit shapes to final decision

**Key Training Terminology**:
- **Epoch**: One complete pass over the entire 60,000-image training set
- **Batch**: Groups of 100 images processed together
- **Iteration**: Processing one batch
- **Loss**: Quantifies prediction error
- **Backpropagation**: Computes how each weight contributed to error
- **Optimizer**: Adjusts weights to reduce loss (Adam optimizer used here)

## Dataset

**MNIST**: 70,000 grayscale images of handwritten digits
- 60,000 training images
- 10,000 test images
- 28×28 pixel resolution

## Model Architecture

A compact CNN with:
- **Conv Layer 1**: 1→10 filters, 5×5 kernels
- **Max Pooling**: Reduces 24×24 to 12×12
- **Conv Layer 2**: 10→20 filters, 5×5 kernels
- **Dropout 2D**: Prevents overfitting
- **Max Pooling**: Reduces 8×8 to 4×4
- **Fully Connected 1**: 320→50 neurons
- **Fully Connected 2**: 50→10 neurons (digit classes)

Final flattened dimension: 4×4×20 = 320

## Training Configuration

- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Cross-entropy loss
- **Epochs**: 10
- **Batch Size**: 100
- **Device**: GPU (if available)

## Results

**Final Performance**:
- Test Accuracy: **93.46%** (9,346/10,000 correct)
- Average Test Loss: 1.5286
- Training Loss: 1.5734

TensorBoard visualization shows test accuracy rising from ~86% in early epochs to
93% by convergence.

## Key Insights

A CNN can learn to recognize handwritten digits from raw pixels without manual
engineering. However, models trained on MNIST require inputs preprocessed to match
MNIST characteristics—centered 28×28 white-on-black grayscale images—to perform
reliably on new handwriting samples.

## Tech Stack

- **PyTorch** & **TorchVision**: Model construction, training loops, dataset utilities
- **MNIST Dataset**: 70,000 labeled digit images
- **Adam Optimizer** & **Cross-Entropy Loss**: Training algorithm
- **TensorBoard**: Real-time metric visualization
- **Matplotlib**: Input visualization and prediction display

## References

- MNIST Database: [The MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- TensorBoard: [TensorFlow's Visualization Toolkit](https://www.tensorflow.org/tensorboard)
