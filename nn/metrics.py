import numpy as np
def accuracy(y_pred,y_true):
    #Gets the index (label) of the max value per sample
    predictions = np.argmax(y_pred, axis=1)
    # If targets are one-hot encoded - convert them
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
    # True evaluates to 1; False to 0
    accuracy = np.mean(predictions == y_true)
    return accuracy
