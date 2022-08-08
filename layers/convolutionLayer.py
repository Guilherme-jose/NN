from layers.layer import layer
from scipy import signal
import activationFunctions
import numpy as np
from weightInit import he

class kernelLayer(layer):
    def __init__(self, inputShape, kernelSize, kernelDepth, activation=activationFunctions.leakyRelu, activationD=activationFunctions.leakyReluD, pad=0, mode='valid') -> None:
        self.inputShape = inputShape #3 dimensions
        self.kernelSize = kernelSize
        self.kernelDepth = kernelDepth
        inputDepth, inputHeight, inputWidth = inputShape
        self.inputDepth = inputDepth
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        if(mode=='valid'):
            self.outputShape = (kernelDepth, inputHeight - kernelSize + 1, inputWidth - kernelSize + 1)
        else:
            self.outputShape = (kernelDepth, inputHeight, inputWidth)
        self.kernelShape = (kernelDepth, inputDepth, kernelSize, kernelSize)
        self.initWeights()
        self.initBias()
        self.pad = pad
        self.actFunc = activation
        self.actFuncDerivative = activationD
        self.mode = mode
        if(self.mode == 'same'):
            self.topPad = (self.kernelShape[2]-1)//2
            self.bottomPad = (self.kernelShape[2])//2
            self.leftPad = (self.kernelShape[3]-1)//2
            self.rightPad = (self.kernelShape[3])//2
        
    def reinit(self) -> None:
        self.initWeights()
        self.initBias()
    
    def backPropagation(self, input, gradient):
        gradient = gradient *  self.actFuncDerivative(self.output)
        input_gradient = np.zeros(self.inputShape)
        
        for i in range(self.kernelDepth):
            for j in range(self.inputDepth):
                if(self.mode == 'same'):
                    input_gradient[j] += signal.convolve2d(gradient[i], self.weights[i, j], "same")
                else:
                    input_gradient[j] += signal.convolve2d(gradient[i], self.weights[i, j], "full")
        return input_gradient
    
    def deltaWeights(self, input, gradient):
        gradient = gradient *  self.actFuncDerivative(self.output)
        kernels_gradient = np.zeros(self.kernelShape)
        for i in range(self.kernelDepth):
            for j in range(self.inputDepth):
                if(self.mode == 'same'):
                    paddedInput = np.pad(input, ((0,0),(self.topPad, self.bottomPad), (self.leftPad, self.rightPad)))
                    kernels_gradient[i, j] = signal.correlate2d(paddedInput[j], gradient[i], "valid")
                else:
                    kernels_gradient[i, j] = signal.correlate2d(input[j], gradient[i], "valid")
        return kernels_gradient
    
    def updateWeights(self, delta):
        self.weights -= self.learningRate * delta
    
    def updateBias(self, gradient):
        self.bias = np.subtract(self.bias, self.learningRate * gradient)
    
    def forward(self, input):
        self.output = np.copy(self.bias)
        for i in range(self.kernelDepth):
            for j in range(self.inputDepth):
                self.output[i] += signal.correlate2d(input[j], self.weights[i, j], self.mode)
                
        self.output = self.actFunc(self.output)
        return self.output

    def initWeights(self):
        self.weights = he(self.kernelShape, self.inputDepth * self.inputHeight * self.inputWidth)
            
    def initBias(self):
        self.bias = he(self.outputShape, self.inputDepth * self.inputHeight * self.inputWidth)
        
    