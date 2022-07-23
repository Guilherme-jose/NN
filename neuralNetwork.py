from imghdr import tests
import math
import random
from tkinter import W
import numpy as np
import pygame
from layers.convolutionLayer import kernelLayer
import activationFunctions
from layers.layer import layer
import lossFunctions
from layers.maxPoolLayer import maxPoolLayer
from layers.reshapeLayer import reshapeLayer
import re
from ast import literal_eval

class NeuralNetwork:
    learningRate = 0.2
    layerList = []
    
    def __init__(self, inputShape):
        self.inputShape = inputShape
        
    def reinit(self):
        for i in self.layerList:
            i.reinit()
        
    def addDenseLayer(self, size, activationFunction=activationFunctions.sigmoid, activationFunctionD=activationFunctions.sigmoidD):
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
        outputMatrix = np.array(input, ndmin=2)
        for it in range(len(self.layerList)):
            outputMatrix = self.layerList[it].forward(outputMatrix)
        
        return outputMatrix.T.tolist()
        
    def train(self, inputSet, outputSet, epochs, mode="", testSamples=0, testSet=None, testOutput=None, batchSize=0):
        iterations = 0
        if(batchSize == 0): batchSize = 1
        for epoch in range(epochs):
            error = 0
            itError = 0
            for batch in range(len(inputSet) // batchSize):
                gradient = 0
                for k in range(batchSize):
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
                    
                    gradient += lossFunctions.mse_prime(outputTarget, outputMatrix)
                    error += lossFunctions.mse(outputTarget, outputMatrix)
                    itError += lossFunctions.mse(outputTarget, outputMatrix)
                        
                    iterations += 1
                    if(iterations%1000==0): 
                        itError /= len(inputSet)
                        print("iterations:", iterations, "error:", itError)
                        itError = 0
                        if(mode == "classifier"):
                            if(testSet == None):
                                testSet = inputSet
                                testOutput = outputSet
                            self.testClassifier(testSet, testOutput, testSamples)
                gradient /= batchSize
                for i in range(len(self.layerList)):
                        gradient = self.layerList[len(self.layerList) - 1 - i].backPropagation(outputList[len(outputList )- 2 - i], outputList[len(outputList) - 1 - i], gradient)
            error /= len(inputSet)
            print("epoch:", epoch, "/",epochs, " -  error:", error)
            self.dumpWeights()
    
    def testClassifier(self, testSet, testOutput, testSamples=0):
        count = 0
        for i in range(testSamples): 
            j = random.randrange(len(testSet) - 1)
            if(np.argmax(self.guess(testSet[j])) == np.argmax(testOutput[j]).T):
                count += 1
        print(count*100/testSamples, "% " + "over " + str(testSamples) + (" tests"))

    def dumpWeights(self):
        file = open("w/weights" + str(random.randrange(0,999999999)) + ".txt", "w")
        
        with np.printoptions(threshold=np.inf):
            for i in self.layerList:
                file.write(str(i.weights))
                file.write("|\n")

        
    def loadWeights(self, id):
        file = open("w/" + id + ".txt", "r")
        text = file.read()
        file.close()
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace("\n", "")
        
        text = text.split("|")
        
        j = 0 
        for i in self.layerList:
            w = text[j].split()
            for k in w :
                k = float(k)
            i.weights = np.array(w, ndmin=2, dtype=float).T.reshape(i.weights.shape)
            print(i.weights.shape)
            j += 1