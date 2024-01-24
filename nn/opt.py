import numpy as np

class Optimizer_SGD():

    def __init__(self, network, learning_rate = 1., decay = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.network = network
        self.decay = decay
        self.iterations = 0

    def learning_rate_decay(self):
        self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def step(self):

        for layer in self.network.layers:
            if hasattr(layer, 'weights'):    
                layer.weights += -self.current_learning_rate * layer.dweights.T
                layer.biases += -self.current_learning_rate * layer.dbiases[0]
        
        if self.decay:
            self.learning_rate_decay()
            self.iterations += 1