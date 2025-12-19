MNIST Digit Recognizer
A desktop application for real-time handwritten digit recognition using a Convolutional Neural Network (CNN) and a GUI based on Tkinter. The system captures user input via a drawing canvas, preprocesses the image to match MNIST dataset specifications, and performs inference using a model trained with PyTorch.

Project Overview
This project implements an end-to-end deep learning pipeline, including:

Model Training: A CNN trained on the MNIST dataset with data augmentation (random rotation, affine transformations) to improve generalization.

Inference Engine: Integration of the trained model into a Python application.

Preprocessing Logic: An algorithm to center and crop user drawings to resolve domain shift issues between high-resolution canvas input and low-resolution training data (28x28 pixels).

Key Features
Interactive Interface: Tkinter-based canvas allowing users to draw digits with a mouse.

Smart Preprocessing: Automatic bounding box detection, cropping, centering, and resizing of the input image to ensure consistent input for the neural network.

Convolutional Neural Network: A custom PyTorch architecture achieving approximately 99% accuracy on the test set.

Confidence Scoring: Displays the predicted class and the softmax probability associated with the prediction.

Technical Stack
Language: Python 3.10+

Deep Learning: PyTorch, Torchvision

GUI: Tkinter

Image Processing: Pillow (PIL), NumPy

Model Architecture
The model (obrazki class) consists of two main blocks:

Feature Extractor (Encoder):

Conv2d (1 input channel, 32 output channels, kernel=3, padding=1) -> BatchNorm -> ReLU -> MaxPool(2)

Conv2d (32 input channels, 64 output channels, kernel=3, padding=1) -> BatchNorm -> ReLU -> MaxPool(2)

Classifier:

Flatten layer

Linear (3136 -> 512) -> ReLU -> Dropout(0.5)

Linear (512 -> 128) -> ReLU -> Dropout(0.5)

Linear (128 -> 10 output classes)
