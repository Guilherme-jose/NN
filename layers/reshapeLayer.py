import numpy as np
import activationFunctions
from layers.layer import layer

class reshapeLayer(layer):
    def __init__(self, inputShape, outputShape, activation=activationFunctions.tanh, activationD=activationFunctions.tanhD) -> None:
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.size = outputShape[0]
        
    def reinit(self) -> None:
        pass
    
    def backPropagation(self, input, gradient):
        r = np.reshape(gradient, self.inputShape)
        return r
        
    def forward(self, input):
        self.output = np.reshape(input, self.outputShape)
        return self.output