from math import sqrt
import numpy as np

def signedUniform(shape):
    return -1 + 2 * np.random.rand(*shape)

def xavier(shape, inputSize):
    out = np.random.rand(*shape)
    lower, upper = -(1.0 / sqrt(inputSize)), (1.0 / sqrt(inputSize))
    out = lower + out * (upper - lower)
    return out

def normXavier(shape, inputSize, outputSize):
    out = np.random.rand(*shape)
    lower, upper = -(sqrt(6.0) / sqrt(inputSize + outputSize)), (sqrt(6.0) / sqrt(inputSize + outputSize))
    out = lower + out * (upper - lower)
    return out

def he(shape, inputSize):
    out = np.random.randn(*shape)
    r = sqrt(2.0 / inputSize)
    out *= r
        
    return out