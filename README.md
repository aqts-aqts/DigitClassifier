# Handwritten Digit Classifier

This repository contains a from-scratch Convolutional Neural Network (CNN) for handwritten digit classification using NumPy. The model is trained and tested on the MNIST dataset.

## Features

- Implementation of a CNN from scratch using NumPy
- Training and evaluation on the MNIST dataset
- No external deep learning libraries used

## Requirements

- Python 3.x
- NumPy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/aqts-aqts/DigitClassifier.git
    cd DigitClassifier
    ```

2. Install the required packages:
    ```bash
    pip install numpy scikit-learn scipy
    ```

## Usage

1. Train the model:
    ```bash
    python train.py
    ```

2. Evaluate the model:
    ```bash
    python test.py
    ```

## Project Structure

- `train.py`: Script to train the CNN model.
- `evaluate.py`: Script to evaluate the trained model.
- `visualize.py`: Script to visualize the results.
- `model.py`: Contains the CNN model implementation.
- `utils.py`: Utility functions for data processing and model operations.
- `data/`: Directory to store the MNIST dataset.
- `weights/`: Stores the trained weights.

## Acknowledgements

- The MNIST dataset is provided by Yann LeCun and can be found [here](http://yann.lecun.com/exdb/mnist/).