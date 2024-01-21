import numpy as np

class ReLU():
    def __init__(self):
        pass
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
        return self.output
