import numpy as np
from scipy.special import expit
from typing import Union
from load import load_image

rate = 0.1

def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return expit(x)

def sigmoid_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    s = sigmoid(x)
    return s * (1 - s)

class Neuron:
    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = weights
        self.bias = bias
        self.z = None
        self.value = None
        self.inputs = None
    
    # Propagate forward
    def forward(self, inputs: np.ndarray) -> float:
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.value = sigmoid(self.z)
        return self.value
    
    # Propagate backward
    def backward(self, error: float, learning_rate: float) -> np.ndarray:
        # a - activation, z - weighted sum, w - weights, b - bias, C - cost
        dC_da = error
        dC_dz = dC_da * sigmoid_derivative(self.z)
        dC_dw = dC_dz * self.inputs
        dC_db = dC_dz

        self.weights -= dC_dw * learning_rate
        self.bias -= dC_db * learning_rate

        dC_da_prev = dC_dz * self.weights
        return dC_da_prev

class Layer:
    def __init__(self, prev_neurons: int, neurons: int):
        self.neurons = [Neuron(np.random.randn(prev_neurons), np.random.randn()) for _ in range(neurons)]
    
    # Propagate forward
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.array([neuron.forward(inputs) for neuron in self.neurons])
    
    # Propagate backward
    def backward(self, errors: np.ndarray) -> np.ndarray:
        prev_error = np.zeros(self.neurons[0].weights.shape)
        for neuron, error in zip(self.neurons, errors):
            prev_error += neuron.backward(error, learning_rate=rate)
        return prev_error

def propagate_forward(layers: list[Layer], inputs: np.ndarray) -> np.ndarray:
    for layer in layers:
        inputs = layer.forward(inputs)
    return inputs

def propagate_backward(layers: list[Layer], errors: np.ndarray) -> np.ndarray:
    for layer in reversed(layers):
        errors = layer.backward(errors)
    return errors

hidden_layers = 2
neurons_per_layer = 16
input_size = 784
output_size = 10

# Initialize layers
layers = [Layer(input_size, neurons_per_layer), *[Layer(neurons_per_layer, neurons_per_layer) for _ in range(hidden_layers - 1)], Layer(neurons_per_layer, output_size)]

# Load weights and biases
for i, layer in enumerate(layers):
    for j, neuron in enumerate(layer.neurons):
        neuron.weights = np.load(f'weights/layer_{i}_neuron_{j}_weights.npy')
        neuron.bias = np.load(f'weights/layer_{i}_neuron_{j}_bias.npy')

# Load testing data
test = load_image('data/test.png')

# Propagate forward through all testing data after training to test
inputs = test
labels = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
output = propagate_forward(layers, inputs)
cost = np.sum((output - labels) ** 2)
print('Network outputs:', output)
print('Guess:', np.argmax(output))
print('Actual:', np.argmax(labels))
print('Cost:', cost)