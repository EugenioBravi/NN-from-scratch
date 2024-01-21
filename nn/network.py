from typing import Type
from layer import Layer
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
        if type(input) != list:
            raise Exception('The input should be a list')
        layers_input = input
        for layer in self.layers:
            layers_input = layer.forward(layers_input)
        return layers_input