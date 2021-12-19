# import libraries
import matplotlib.pyplot as plt
import numpy as np

from NeuralNetwork import NeuralNetwork 
from MNIST import MNIST

dataset = MNIST()
neural_network = NeuralNetwork(28 * 28, 100, 10, 0.3)

# loop through training set
for i in range(len(dataset.train_set)):

    print(f"epoch: {i:04}")

    # initialize variables
    record = dataset.train_set[i]
    all_values = record.split(',')

    # preprocess data
    scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # expected output
    targets = np.zeros(neural_network.onodes) + 0.01
    targets[int(all_values[0])] = 0.99

    # train the neural network
    neural_network.train(scaled_input, targets)

print("finished training!")

# test the neural network
all_values = dataset.test_set[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28, 28))

# plot image array
plt.figure()
plt.imshow(image_array, cmap="Greys", interpolation="None")
plt.show()

result = neural_network.predict(np.asfarray(all_values[1:]))
print(f"prediction: {np.argmax(result)} confidence: {np.max(result)}")