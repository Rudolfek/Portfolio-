# MNIST Digit Recognizer

A real-time handwritten digit recognition application built with Python, PyTorch, and Tkinter. The project implements an end-to-end deep learning pipeline, ranging from training a Convolutional Neural Network (CNN) on the MNIST dataset to deploying the model in an interactive GUI.

## Project Overview

The primary goal of this project is to demonstrate the practical deployment of a computer vision model. It addresses the common "domain shift" problem where models trained on pre-processed datasets fail on raw real-world input.

Key components:
1.  **Deep Learning Model:** A custom CNN architecture trained to classify digits (0-9).
2.  **Data Augmentation:** The training process includes random rotations and affine transformations to improve model robustness.
3.  **Preprocessing Pipeline:** A custom algorithm that detects the bounding box of user input, centers the digit, adds padding, and resizes it to 28x28 pixels to match the MNIST data distribution.

## Technical Architecture

### Model Structure
The neural network (`obrazki` class) utilizes a sequential architecture with two main blocks:

**1. Feature Extractor (Encoder):**
* **Layer 1:** Conv2d (1 in, 32 out, kernel=3, padding=1) → BatchNorm → ReLU → MaxPool(2)
* **Layer 2:** Conv2d (32 in, 64 out, kernel=3, padding=1) → BatchNorm → ReLU → MaxPool(2)

**2. Classifier:**
* **Flatten:** Converts 2D feature maps to a 1D vector.
* **Fully Connected:** Linear (3136 → 512) → ReLU → Dropout(0.5)
* **Fully Connected:** Linear (512 → 128) → ReLU → Dropout(0.5)
* **Output Layer:** Linear (128 → 10 classes)

### Technologies
* **Language:** Python 3.10+
* **Machine Learning:** PyTorch, Torchvision
* **GUI Framework:** Tkinter
* **Image Processing:** Pillow (PIL), NumPy
