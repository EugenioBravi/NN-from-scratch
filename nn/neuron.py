import random

class Neuron():

    def __init__(self,n_weights: int,bias : float = round(random.uniform(-1,1),2) ) -> None:

        if type(n_weights) != int or n_weights <= 0 or n_weights >20 :
            raise Exception('The param n_weights should be an int beetwen 1 and 20')

        self.weights = [round(random.uniform(-1,1),2) for _ in range(n_weights)] 
        self.bias = bias