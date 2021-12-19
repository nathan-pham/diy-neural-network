# import libraries 
import matplotlib.pyplot as plt
import numpy as np

from typing import List

# MNIST class
class MNIST:

    # constructor
    def __init__(self):

        # initialize datasets
        self.train_set = MNIST.read_csv("dataset/mnist_train_100.csv")
        self.test_set = MNIST.read_csv("dataset/mnist_test_10.csv")

    # display a csv row as an image
    def display_image(self, i: int = 0):

        # initialize variables
        all_values = self.train_set[i].split(',')
        image_array = np.asfarray(all_values[1:]).reshape((28, 28))

        # plot image array
        plt.figure()
        plt.imshow(image_array, cmap="Greys", interpolation="None")
        plt.show()

        return image_array

    # read a csv file
    @staticmethod
    def read_csv(filename: str) -> List[str]:
        with open(filename, 'r') as handle:
            return handle.readlines()
