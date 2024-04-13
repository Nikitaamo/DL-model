# Deep Learning Model Comparison

# Introduction

This project demonstrates the use of deep learning frameworks TensorFlow/Keras and PyTorch to classify non-linearly separable data generated by the make_moons function. This comparative analysis aims to explore the effectiveness and operational differences between these two popular frameworks in handling a binary classification problem.

## Model Details

### TensorFlow/Keras
The TensorFlow/Keras model utilizes a straightforward neural network architecture comprising three layers:

An input layer with two features corresponding to the two dimensions of the make_moons data.
Two hidden layers with 10 neurons each, using ReLU activation to introduce non-linearity.
A softmax output layer that classifies the input into two categories.
This model was optimized using the adam optimizer and trained to minimize categorical cross-entropy, reflecting a focus on robust classification performance.

### PyTorch
The PyTorch implementation mirrors the complexity of the Keras model:

The network structure consists of an input layer, two hidden layers with 10 neurons each, and a softmax output layer for binary classification.
It employs the ReLU activation function for non-linear transformations between layers.
Training involves the Negative Log-Likelihood Loss and the Adam optimizer, aligning with the model's requirement to effectively generalize over the non-linear data distribution.

## Installation and Setup

Ensure Python 3 and the necessary libraries are installed. Install the required packages with the following command: pip install -r requirements.txt

Run the model by executing the src.py script: python src.py

## Logging

Logging is integral to our project, capturing detailed outputs during the training and evaluation phases. The log.txt file includes epoch-wise accuracy and loss metrics for both frameworks, allowing for an in-depth performance analysis.

## Visualization

The project generates a visualization IMG.png that illustrates the accuracy trends across training epochs for both models. This visual aids in comparing how quickly each model converges to its optimal accuracy.

## Comparative Analysis

Through this project, we aim to not only demonstrate the application of deep learning models to a simple dataset but also to provide insights into how different frameworks can be utilized effectively. TensorFlow/Keras is praised for its ease of use and extensive support, making it ideal for beginners and projects requiring rapid development. On the other hand, PyTorch offers more granular control over model training, which is advantageous for research purposes and when precise model behavior adjustments are necessary.

## Conclusion

This project serves as a foundation for understanding and applying TensorFlow/Keras and PyTorch frameworks in practical deep learning tasks. The insights gained from the comparative analysis will help guide future model choices and framework applications depending on the specific needs of the task at hand.
