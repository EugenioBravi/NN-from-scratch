import numpy as np

class Optimizer_SGD():

    def __init__(self, network, learning_rate=0.3):
        self.learnig_rate = learning_rate
        self.network = network

    def step(self):
        for layer in self.network.layers:
            if hasattr(layer, 'weights'):    
                layer.weights += -self.learnig_rate * layer.dweights.T
                layer.biases += -self.learnig_rate * layer.dbiases[0]