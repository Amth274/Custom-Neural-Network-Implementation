# Custom Neural Network Implementation

## Introduction

This project is a custom implementation of a neural network from scratch, designed for multi-class classification. The neural network supports various optimization algorithms, including Stochastic Gradient Descent (SGD), Momentum, Adagrad, RMSprop, and Adam. The implementation covers forward propagation, backpropagation, and parameter updates, with options for mini-batch and full-batch gradient descent.

## Features

- **Feedforward Neural Network**: Custom implementation from scratch.
- **Activation Functions**: ReLU (Rectified Linear Unit) and Sigmoid.
- **Optimization Algorithms**: SGD, Momentum, Adagrad, RMSprop, and Adam.
- **Training**: Mini-batch and full-batch gradient descent.
- **Loss Function**: Cross-entropy loss for multi-class classification.
- **Model Architecture Visualization**: Displays the architecture of the network.

## Prerequisites

- Python 3.7 or higher
- NumPy library

You can install NumPy using pip:
```bash
pip install numpy
```

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/custom-neural-network.git
cd custom-neural-network
```
Install dependencies
```python
pip install numpy
```

## Usage
Prepare the Data:

Prepare your dataset in a suitable format (e.g., NumPy arrays).
Update the data loading and preprocessing code as needed.
Instantiate and Train the Model:

Open main.py and configure the neural network parameters and training settings. Example usage:

```python 
import numpy as np
from neural_network import NeuralNetwork, Optimizer

# Example data
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.randint(0, 3, 100)  # 100 labels for 3 classes

# Define the model architecture
layers = [10, 64, 32, 3]  # Input layer, two hidden layers, output layer

# Instantiate the neural network and optimizer
nn = NeuralNetwork(layers)
nn.optimizer = Optimizer(optimizer_type='adam', learning_rate=0.01)

# Train the network
nn.train(X, y, epochs=1000, batch_size=32, learning_rate=0.01)
```
## Contributing 
If you would like to contribute to this project, please follow these steps:

Fork the repository: Click the "Fork" button on the top right of this page.

Create a new branch:
```bash
git checkout -b feature/YourFeature
```

Commit your changes:
```bash
git commit -am 'Add new feature'
```

Push to the branch:
```bash
git push origin feature/YourFeature
```


## Contact
For questions or feedback, you can reach out to me:

-Name: Aum Thaker

-Email: 225100006@iitdh.ac.in

-GitHub: github.com/Amth274


Feel free to connect or follow me for updates and new projects!




