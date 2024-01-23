import random
import numpy as np
from typing import Type

class Neuron():

    def __init__(self,n_weights: int,bias : bool = False ) -> None:

        if type(n_weights) != int or n_weights <= 0 or n_weights >20 :
            raise Exception('The param n_weights should be an int beetwen 1 and 20')

        self.weights = [random.uniform(-1,1) for _ in range(n_weights)] 
        if bias == True:
            self.bias = random.uniform(-1,1)
        else:
            self.bias = 0

class Layer():
    def __init__(self,n_weights:int,n_neurons:int) -> None:
        
        if type(n_neurons) != int or n_neurons <= 0 or n_neurons >20 :
            raise Exception('The param n_neurons should be an int beetwen 1 and 20')
        if type(n_weights) != int or n_weights <= 0 or n_weights >20 :
            raise Exception('The param n_weights should be an int beetwen 1 and 20')
        
        self.neurons=[Neuron(n_weights) for _ in range(n_neurons)]
        self.n_weights = n_weights
        self.weights = self.get_weights() 
        self.bias = self.get_bias()
        
    def forward(self,inputs:[[float]]) -> [float]:

        if len(inputs[0]) != self.n_weights:
            raise Exception(f'The inputs should be a list of len(weight)={self.n_weights}, {inputs[0]}')
        
        self.inputs = inputs
        output = np.dot(inputs,self.weights.T) + self.bias
        return output
    
    #Returns the traspose array of weights of the layer
    def get_weights(self)->[[float]]:
        weights = []
        for neuron in self.neurons:
            weights.append(neuron.weights)
        return np.array(weights)
    
    #Returns the array of bias of the layer
    def get_bias(self)->[float]:
        bias = []
        for neuron in self.neurons:
            bias.append(neuron.bias)
        return np.array(bias)
    
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues, self.get_weights().T)
        return self.dinputs

class Network():
    def __init__(self) -> None:
        self.layers = []
        
    #Allows adding one layer at a time at the end of the sequence of layers OR to replace the whole network of layers 
    def add_layer(self,layer:Type[Layer] or [Type[Layer]])->None:
        if type(layer) == list:
            self.layers = layer
        else:
            self.layers.append(layer)

    def forward(self,input:[float]) -> [float]:
        if type(input) != list and type(input) != np.ndarray:
            raise Exception('The input should be a list')
        layers_input = input
        for layer in self.layers:
            layers_input = layer.forward(layers_input)
        return layers_input
    
    def backward(self,dvalues):
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)