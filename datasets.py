import numpy as np

def spiral_dataset(samples,classes):
    np.random.seed(0)
    N = samples # number of points per class
    D = 2 # dimensionality
    K = classes # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X,y

def data_split(X,y,p):
    #splits the data in a percentage p (0 > p < 1)
    if p <= 0 or p >=1:
        raise Exception('p value should be between 0 and 1') 
    #Shuffles the data
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    #Splits the data 
    split = round(len(X)* p)
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]
    return X_train, X_test, y_train, y_test