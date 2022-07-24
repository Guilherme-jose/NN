import random
import numpy as np
import activationFunctions

class layer:
    def activation():
        pass
    
    def activationDerivative():
        pass
    
    weights = np.array([])
    bias = np.array([])
    learningRate = 0.1
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
        delta = gradient@input.T
        error = self.weights@gradient
        self.weights = np.subtract(self.weights, self.learningRate * delta.T)
        self.bias =  np.subtract(self.bias, self.learningRate * gradient)
        return error
    
    def initWeights(self):
        temp = []
        for i in range(self.inputShape[0]):
            temp2 = []
            for j in range(self.outputShape[0]):
                temp2.append(random.uniform(-1,1))
            temp.append(temp2)
        self.weights = np.array(temp, ndmin=2)
            
    def initBias(self):
        temp = []
        for j in range(self.outputShape[0]):
            temp.append(random.uniform(-1,1))
        self.bias = np.array(temp, ndmin=2).T
    