import random
import numpy as np
import activationFunctions
import weightInit
from layers.layer import layer

class dense(layer):
    def activation():
        pass
    
    def activationDerivative():
        pass
    
    weights = np.array([])
    bias = np.array([])
    learningRate = 0.02
    inputSize = 0

    def __init__(self, inputShape, outputShape, activation=activationFunctions.sigmoid, activationD=activationFunctions.sigmoid) -> None:
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.initWeights()
        self.initBias()
        self.actFunc = activation
        self.actFuncDerivative = activationD

    def reinit(self) -> None:
        self.initWeights()
        self.initBias()
    
    #takes input as matrix, for use inside the network
    def forward(self, input):
        self.output = self.actFunc(   self.weights.T@input + self.bias    )
        return self.output
    
    def backPropagation(self, input, gradient):
        gradient = gradient * self.actFuncDerivative(self.output)
        error = self.weights@gradient
        return error
    
    def deltaWeights(self, input, gradient):
        gradient = gradient * self.actFuncDerivative(self.output)
        delta = gradient@input.T
        return delta.T
    
    def updateWeights(self, gradient):
        self.weights = np.subtract(self.weights, self.learningRate * gradient)
    
    def updateBias(self, gradient):
        self.bias = np.subtract(self.bias, self.learningRate * gradient)
        
    def initWeights(self):
        self.weights = weightInit.normXavier((self.inputShape[0], self.outputShape[0]), self.inputShape[0], self.outputShape[0])
            
    def initBias(self):
        self.bias = weightInit.normXavier((self.outputShape[0], 1), self.inputShape[0], self.outputShape[0])
    