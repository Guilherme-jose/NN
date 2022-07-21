from layers.layer import layer
from scipy import signal
import activationFunctions
import numpy as np

class kernelLayer(layer):
    def __init__(self, inputShape, kernelSize, kernelDepth, activation=activationFunctions.sigmoid, activationD=activationFunctions.sigmoidD, pad=0) -> None:
        self.inputShape = inputShape #3 dimensions
        self.kernelSize = kernelSize
        self.kernelDepth = kernelDepth
        inputDepth, inputHeight, inputWidth = inputShape
        self.inputDepth = inputDepth
        self.outputShape = (kernelDepth, inputHeight - kernelSize + 1, inputWidth - kernelSize + 1)
        self.kernelShape = (kernelDepth, inputDepth, kernelSize, kernelSize)
        self.initWeights()
        self.initBias()
        self.pad = pad
        self.actFunc = activation
        self.actFuncDerivative = activationD
        
    def reinit(self) -> None:
        self.initWeights()
        self.initBias()
    
    def backPropagation(self, input, output, gradient):
        gradient = gradient *  self.actFuncDerivative(output)
        kernels_gradient = np.zeros(self.kernelShape)
        input_gradient = np.zeros(self.inputShape)

        for i in range(self.kernelDepth):
            for j in range(self.inputDepth):
                kernels_gradient[i, j] = signal.correlate2d(input[j], gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(gradient[i], self.weights[i, j], "full")

        self.weights -= self.learningRate * kernels_gradient
        self.bias -= self.learningRate * gradient
        return input_gradient
    
    def forward(self, input):
        output = np.copy(self.bias)
        for i in range(self.kernelDepth):
            for j in range(self.inputDepth):
                output[i] += signal.correlate2d(input[j], self.weights[i, j], "valid")
                
        output = self.actFunc(output)
        return output

    def initWeights(self):
        self.weights = np.random.randn(*self.kernelShape)
            
    def initBias(self):
        self.bias = np.random.randn(*self.outputShape)