import numpy as np

class ReLU():
    def __init__(self):
        pass
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
        return self.output

class softmax():
    def forward(self,inputs):
        #Get unnormalized probabilities

        #" inputs - np.max(inputs) " to prevent the exponential function from overflowing, with Softmax, thanks to the normalization, we can subtract any value from all of the inputs, and it will not change the output

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
         # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        
        self.output = probabilities
        return self.output