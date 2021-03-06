# import libraries
from scipy.special import expit
import numpy as np

from typing import List

# NeuralNetwork class
class NeuralNetwork:

    # constructor
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float) -> None:

        # initialize variables
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        # initialize weights
        self.wih = np.random.normal(0.0, self.inodes ** -0.5, (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, self.hnodes ** -0.5, (self.onodes, self.hnodes))

        # initialize activation function
        self.activation_function = lambda x : expit(x)

    # convert an array to a 2d array
    @staticmethod
    def to_array(inputs: List[float]) -> np.array:
        return np.array(inputs, ndmin=2).T

    # train the neural network
    def train(self, inputs_list: List[float], targets_list: List[float]) -> None:

        # convert inputs and targets to 2d arrays
        inputs = NeuralNetwork.to_array(inputs_list)
        targets = NeuralNetwork.to_array(targets_list)

        # signals -> hidden layer -> signals
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # signals -> output layer -> signals
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # calculate errors
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # update weights for hidden -> output layer
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))

        # update weights for input -> hidden layer
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))

    # query the neural network
    def predict(self, _inputs: List[float]) -> np.array:

        # convert inputs to 2d array
        inputs = NeuralNetwork.to_array(_inputs)

        # signals -> hidden layer -> signals
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # signals -> output layer -> signals
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs