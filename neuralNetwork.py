from imghdr import tests
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
    
    def __init__(self, inputShape):
        self.inputShape = inputShape
        
    def reinit(self):
        for i in self.layerList:
            i.reinit()
        
    def addDenseLayer(self, size, activationFunction=activationFunctions.tanh, activationFunctionD=activationFunctions.tanhD):
        prevSize = self.inputShape[0]
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
        inputMatrix = np.array(input, ndmin=2)
        
        outputMatrix = inputMatrix
        for it in range(len(self.layerList)):
            outputMatrix = self.layerList[it].forward(outputMatrix)
            
        return outputMatrix.T.tolist()
        
    def train(self, inputSet, outputSet, epochs, mode="", testSamples=10):
        self.testSamples = testSamples
        iterations = 0
        for epoch in range(epochs):
            for k in range(len(inputSet)):
                j = random.randrange(0, len(outputSet))
                
                outputTarget = np.zeros(self.inputShape)
                inputMatrix = np.zeros(self.inputShape)
                outputList = []

                outputTarget = np.array(outputSet[j], ndmin=2).T
                inputMatrix = np.array(inputSet[j], ndmin=2)
                
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
    
    def testClassifier(self, trainingSet, trainingOutput, testSet=None, testOutput=None):
        if testSet == None:
            testSet = trainingSet
            testOutput = trainingOutput
        count = 0
        for i in range(self.testSamples): 
            j = random.randrange(len(trainingSet) - 1)
            if(np.argmax(self.guess(testSet[j])) == np.argmax(testOutput[j]).T):
                count += 1
        print(count*100/self.testSamples, "% " + "over " + str(self.testSamples) + (" tests"))
        self.dumpWeights()

    def dumpWeights(self):
        file = open("w/weights" + str(random.randrange(0,999999999)) + ".txt", "w")
        for i in self.layerList:
            with np.printoptions(threshold=np.inf):
                file.write(str(i.weights))