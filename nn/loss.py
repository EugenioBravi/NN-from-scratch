import numpy as np
class loss():
    def calculate(self,pred,y_true):
        sample_losses = self.forward(pred,y_true)
        data_loss = np.mean(sample_losses)
        return data_loss
