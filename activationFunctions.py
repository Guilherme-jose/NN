import numpy as np

def sigmoid(x):
    x = np.clip( x, -500, 500 )
    return 1 / (1 + np.exp(-x))
    
def sigmoidD(x):
    x = np.clip( x, -500, 500 )
    return np.multiply(x, 1 - x)
    
def tanh(x):
    return np.tanh(x)

def tanhD(x):
    return (1.0 - np.multiply((x), (x)))

def softmax(x):
    x = np.clip( x, -500, 500 )
    return np.exp(x)/sum(np.exp(x))

def softmax(x):
    x = np.clip( x, -500, 500 )
    return np.exp(x)/sum(np.exp(x))

def relu(x):
    return np.maximum(0.1*x,x)

def reluD(x):
    return np.where(x <= 0, 0, 1)

def leakyRelu(x):
    x = np.clip( x, -500, 500 )
    return np.maximum(0,x)

def leakyReluD(x):
    return np.where(x <= 0, 0.1, 1)