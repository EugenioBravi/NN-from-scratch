import numpy as np
class Loss():
    def calculate(self,pred,y_true):
        #Calculates sample loss
        sample_losses = self.forward(pred,y_true)
        #Calculates mean loss
        data_loss = np.mean(sample_losses)
        return data_loss

class Categorical_cross_entropy(Loss):
    def forward(self,y_pred,y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
            range(samples),
            y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
            y_pred_clipped*y_true,
            axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient to obtain the average gradient per sample
        #Normalizing the gradient by the number of samples is to make the gradient values independent of the size of the dataset
        self.dinputs = self.dinputs / samples
