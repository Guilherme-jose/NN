import math
import random
import numpy as np
import pygame
from layers.convolutionLayer import kernelLayer
import activationFunctions
from layers.layer import layer
import lossFunctions
from layers.maxPoolLayer import maxPoolLayer
from layers.reshapeLayer import reshapeLayer

class NeuralNetwork:
    learningRate = 0.2
    layerList = []
    batchSize = 1.0 
    
    def __init__(self, input, inputY=1):
        self.inputSize = input
        self.inputSizeT = inputY
        
    def reinit(self):
        for i in self.layerList:
            i.reinit()
        
    def addDenseLayer(self, size, activationFunction=activationFunctions.tanh, activationFunctionD=activationFunctions.tanhD):
        prevSize = self.inputSize
        if(len(self.layerList) > 0):
            prevSize = self.layerList[len(self.layerList) - 1].outputShape[0]
        l = layer((prevSize, 1), (size, 1), activationFunction, activationFunctionD)
        self.layerList.append(l)
        
    def addConvLayer(self, inputShape, kernelSize, kernelDepth=1, activation=activationFunctions.sigmoid, activationD=activationFunctions.sigmoidD):
        l = kernelLayer(inputShape, kernelSize, kernelDepth, activation, activationD)
        self.layerList.append(l)
        
    def addReshapeLayer(self, inputShape, outputShape):
        l = reshapeLayer(inputShape, outputShape)
        self.layerList.append(l)
        
    def addMaxPoolLayer(self, inputShape, outputShape):
        l = maxPoolLayer(inputShape, outputShape)
        self.layerList.append(l)
        
    def guess(self, input):
        inputMatrix = np.array(input, ndmin=2).T
        
        outputMatrix = inputMatrix
        for it in range(len(self.layerList)):
            outputMatrix = self.layerList[it].forward(outputMatrix)
            
        return outputMatrix.tolist()
        
    def train(self, inputSet, outputSet, epochs, mode=""):
        iterations = 0
        for epoch in range(epochs):
            for k in range(len(inputSet)):
                j = random.randrange(0, len(outputSet))
                
                outputList = []

                outputTarget = np.array(outputSet[j], ndmin=2).T
                inputMatrix = np.array(inputSet[j], ndmin=2).T
                
                outputList.append(inputMatrix)
                outputMatrix = inputMatrix
                
                for it in range(len(self.layerList)):
                    outputMatrix = self.layerList[it].forward(outputMatrix)
                    outputList.append(outputMatrix)
                
                error = lossFunctions.mse_prime(outputTarget, outputMatrix)
                
                for i in range(len(self.layerList)):
                    error = self.layerList[len(self.layerList) - 1 - i].backPropagation(outputList[len(outputList )- 2 - i], outputList[len(outputList) - 1 - i], error)
                    
                    
                iterations += 1
                if(iterations%1000==0): 
                    print("iterations:", iterations)
                    if(mode == "classifier"):
                        pass
                        self.testClassifier(inputSet, outputSet)
            print("epoch:", epoch)
            if(mode == "classifier"):
                        self.testClassifier(inputSet, outputSet)
    
    def testClassifier(self, trainingSet, trainingOutput):
        t = max(1, math.floor(len(trainingSet)/100))
        count = 0
        for i in range(t): 
            j = random.randrange(len(trainingSet) - 1)
            if(np.argmax(self.guess(trainingSet[j])) == np.argmax(trainingOutput[j])):
                count += 1
        print(count*100/t, "% " + "over " + str(t) + (" tests"))
    