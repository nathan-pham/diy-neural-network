# import libraries
import numpy as np

from NeuralNetwork import NeuralNetwork 
from MNIST import MNIST


dataset = MNIST()
neural_network = NeuralNetwork(28 * 28, 100, 10, 0.3)

# loop through training set
for record in dataset.train_set:

    # initialize variables
    all_values = record.split(',')
    scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(neural_network.onodes) + 0.01

    targets[int(all_values[0])] = 0.99
    neural_network.train(scaled_input, targets)

print(neural_network.predict([1.0, 0.5, -1.0]))