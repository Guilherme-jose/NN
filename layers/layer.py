import random
import numpy as np
import activationFunctions
import weightInit

class layer:
    def activation():
        pass
    
    def activationDerivative():
        pass
    
    weights = np.zeros((0,0))
    bias = np.zeros((0,0))
    learningRate = 0.02
    inputSize = 0

    def __init__(self) -> None:
        pass

    def reinit(self) -> None:
        pass
    
    #takes input as matrix, for use inside the network
    def forward(self, input):
        pass
    
    def backPropagation(self, input, gradient):
        return np.zeros(0)
    
    def deltaWeights(self, input, gradient):
        return np.zeros(0)
    
    def deltaBias(self, input, gradient):
        return np.zeros(0)
    
    def updateWeights(self, gradient):
        pass
    
    def updateBias(self, gradient):
        pass
    
    def initWeights(self):
        self.weights = weightInit.normXavier((self.inputShape[0], self.outputShape[0]), self.inputShape[0], self.outputShape[0])
            
    def initBias(self):
        self.bias = weightInit.normXavier((self.outputShape), self.inputShape[0], self.outputShape[0])
    