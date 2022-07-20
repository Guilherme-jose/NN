import random
import numpy as np
import activationFunctions

class layer:
    def activation():
        pass
    
    def activationDerivative():
        pass
    
    weights = []
    bias = []
    learningRate = 0.1
    inputSize = 0
    
    def __init__(self, inputShape, outputShape, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
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
        output = self.weights.T@input + self.bias
        output = self.actFunc(output)
        return output
    
    def backPropagation(self, input, output, gradient):
        gradient = gradient * self.actFuncDerivative(output)
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
    
    
class softmaxLayer(layer):
    def activation():
        pass
    
    def activationDerivative():
        pass
    
    weights = []
    bias = []
    learningRate = 0.1
    size = 0
    inputSize = 0
    
    actFunc = activation
    actFuncDerivative = activationDerivative
    
    def __init__(self, inputSize) -> None:
        self.size = inputSize
        self.inputSize = inputSize
        
    def reinit(self) -> None:
        pass
    
    #takes input as matrix, for use inside the network
    def forward(self, input):
        output = activationFunctions.softmax(input)
        return output
    
    def backPropagation(self, input, output, error):
        pass
    
    def findError(self, prevError):
        return prevError


class reshapeLayer(layer):
    def __init__(self, inputShape, outputShape, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.size = outputShape[0]
        
    def reinit(self) -> None:
        pass
    
    def backPropagation(self, input, output, error):
        r = np.reshape(error, self.inputShape)
        return r
        
    def forward(self, input):
        output = np.reshape(input, self.outputShape)
        return output

    