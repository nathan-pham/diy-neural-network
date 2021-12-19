from NeuralNetwork import NeuralNetwork 

neural_network = NeuralNetwork(3, 3, 3, 0.1)

print(neural_network.predict([1.0, 0.5, -1.0]))