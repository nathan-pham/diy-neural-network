# import libraries
import matplotlib.pyplot as plt
import numpy as np

from NeuralNetwork import NeuralNetwork 
from MNIST import MNIST

dataset = MNIST(test=False)
neural_network = NeuralNetwork(28 * 28, 300, 10, 0.2)

# train neural network i times
for i in range(5):

    # loop through training set
    print(f"epoch: {i + 1:04}")
    for record in dataset.train_set:

        # initialize variables
        all_values = record.split(',')

        # preprocess data
        scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(neural_network.onodes) + 0.01
        targets[int(all_values[0])] = 0.99

        # train the neural network
        neural_network.train(scaled_input, targets)

print("finished training!\n")

# test the neural network
scorecard = []
for record in dataset.test_set:

    # initialize variables
    all_values = record.split(',')

    # image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    # plt.figure()
    # plt.imshow(image_array, cmap="Greys", interpolation="None")
    # plt.show()

    # preprocess data
    scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    output = neural_network.predict(scaled_input)
    label = np.argmax(output)

    print(f"prediction: {label} expected: {all_values[0]} confidence: {np.max(output) * 100:.2f}%")
    scorecard.append(1 if int(all_values[0]) == label else 0)

scorecard_array = np.asarray(scorecard)
print(f"performance: {scorecard_array.sum() / scorecard_array.size * 100:.2f}%")