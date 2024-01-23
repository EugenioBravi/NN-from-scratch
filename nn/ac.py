import numpy as np

class ReLU():

    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        #Zero gradient where input values were negetave
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class softmax():
    def forward(self,inputs):
        #Get unnormalized probabilities

        #" inputs - np.max(inputs) " to prevent the exponential function from overflowing, with Softmax, thanks to the normalization, we can subtract any value from all of the inputs, and it will not change the output

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
         # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        
        self.output = probabilities
        return self.output
    
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        return self.dinputs