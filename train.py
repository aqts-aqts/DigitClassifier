import numpy as np
from scipy.special import expit
from sklearn.utils import shuffle
from typing import Union
from load import load_images

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
        
        # For accumulating gradients
        self.dC_dw_accum = np.zeros_like(weights)
        self.dC_db_accum = 0.0
    
    # Propagate forward
    def forward(self, inputs: np.ndarray) -> float:
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.value = sigmoid(self.z)
        return self.value
    
    # Accumulate gradients during backward propagation
    def accumulate_gradients(self, error: float):
        dC_da = error
        dC_dz = dC_da * sigmoid_derivative(self.z)
        dC_dw = dC_dz * self.inputs
        dC_db = dC_dz

        self.dC_dw_accum += dC_dw
        self.dC_db_accum += dC_db
        
        # Return the error propagated to the previous layer
        return dC_dz * self.weights
    
    # Update weights and biases using the average gradients
    def update_parameters(self, batch_size: int, learning_rate: float):
        self.weights -= (self.dC_dw_accum / batch_size) * learning_rate
        self.bias -= (self.dC_db_accum / batch_size) * learning_rate
        
        # Reset the accumulated gradients
        self.dC_dw_accum = np.zeros_like(self.weights)
        self.dC_db_accum = 0.0

class Layer:
    def __init__(self, prev_neurons: int, neurons: int):
        self.neurons = [Neuron(np.random.randn(prev_neurons), np.random.randn()) for _ in range(neurons)]
    
    # Propagate forward
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.array([neuron.forward(inputs) for neuron in self.neurons])
    
    # Accumulate gradients during backward propagation
    def backward(self, errors: np.ndarray) -> np.ndarray:
        prev_error = np.zeros(self.neurons[0].weights.shape)
        for neuron, error in zip(self.neurons, errors):
            prev_error += neuron.accumulate_gradients(error)
        return prev_error
    
    # Update all neurons' parameters after accumulating gradients
    def update_parameters(self, batch_size: int, learning_rate: float):
        for neuron in self.neurons:
            neuron.update_parameters(batch_size, learning_rate)

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

# Load training data
zeros = load_images('data/training/0')
ones = load_images('data/training/1')
twos = load_images('data/training/2')
threes = load_images('data/training/3')
fours = load_images('data/training/4')
fives = load_images('data/training/5')
sixes = load_images('data/training/6')
sevens = load_images('data/training/7')
eights = load_images('data/training/8')
nines = load_images('data/training/9')
print("Loaded images")

labels = {
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0): zeros,
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0): ones,
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0): twos,
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0): threes,
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0): fours,
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0): fives,
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0): sixes,
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0): sevens,
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0): eights,
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 1): nines
}

images_list = []
labels_list = []

for label, images in labels.items():
    for image in images:
        images_list.append(image)
        labels_list.append(label)

images_array = np.array(images_list)
labels_array = np.array(labels_list)
images_array, labels_array = shuffle(images_array, labels_array, random_state=42)
print("Training data loaded")

# Set training parameters
epochs = 100  # Number of epochs
batch_size = 32  # Batch size
learning_rate = 1  # Learning rate

# Train
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    
    for batch_start in range(0, len(images_array), batch_size):
        batch_images = images_array[batch_start:batch_start + batch_size]
        batch_labels = labels_array[batch_start:batch_start + batch_size]
        
        # Accumulate gradients for each image in the batch
        for image, label in zip(batch_images, batch_labels):
            inputs = image
            labels = label

            # Propagate forward
            output = propagate_forward(layers, inputs)

            # Propagate backward
            errors = output - labels
            propagate_backward(layers, errors)
        
        # Update the parameters after processing the entire batch
        for layer in layers:
            layer.update_parameters(batch_size, learning_rate)

    print("Training step completed")

# Save weights and biases if needed
for i, layer in enumerate(layers):
    for j, neuron in enumerate(layer.neurons):
        np.save(f'weights/layer_{i}_neuron_{j}_weights.npy', neuron.weights)
        np.save(f'weights/layer_{i}_neuron_{j}_bias.npy', neuron.bias)

# Load testing data
zeros = load_images('data/testing/0')
ones = load_images('data/testing/1')
twos = load_images('data/testing/2')
threes = load_images('data/testing/3')
fours = load_images('data/testing/4')
fives = load_images('data/testing/5')
sixes = load_images('data/testing/6')
sevens = load_images('data/testing/7')
eights = load_images('data/testing/8')
nines = load_images('data/testing/9')

labels = {
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0): zeros,
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0): ones,
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0): twos,
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0): threes,
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0): fours,
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0): fives,
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0): sixes,
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0): sevens,
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0): eights,
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 1): nines
}

images_list = []
labels_list = []

for label, images in labels.items():
    for image in images:
        images_list.append(image)
        labels_list.append(label)

images_array = np.array(images_list)
labels_array = np.array(labels_list)
images_array, labels_array = shuffle(images_array, labels_array, random_state=42)
print("Testing data loaded")

# Propagate forward through all testing data after training to test
accuracy = 0
for image, label in zip(images_array, labels_array):
    inputs = image
    labels = label
    output = propagate_forward(layers, inputs)
    cost = np.sum((output - labels) ** 2)
    accuracy += np.argmax(output) == np.argmax(labels)
    print('Network outputs:', output)
    print('Guess:', np.argmax(output))
    print('Actual:', np.argmax(labels))
    print('Cost:', cost)
print('Accuracy:', str(accuracy / len(images_array) * 100) + '%')