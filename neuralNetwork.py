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
    layerList = []
    
    def __init__(self, inputShape, lr=0.2):
        self.inputShape = inputShape
        self.lr = lr
        
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
        
        return outputMatrix
        
    def train(self, inputSet, outputSet, epochs, mode="", testSamples=0, testSet=None, testOutput=None, batchSize=1):
        self.iterations = 0
        self.inputSet = inputSet
        self.outputSet = outputSet
        self.mode = mode
        self.testSamples = testSamples
        
        for epoch in range(epochs):
            self.batchError = 0
            self.itError = 0
            for batch in range(len(inputSet) // batchSize):
                gradient = 0
                for k in range(batchSize):
                    j = random.randrange(0, len(outputSet))
                    
                    outputTarget = np.array(outputSet[j], ndmin=2).T
                    inputMatrix = np.array(inputSet[j], ndmin=2)
                    
                    outputMatrix = self.guess(inputMatrix)
                    gradient += self.updateLoss(outputTarget, outputMatrix)
                        
                    self.iterations += 1
                    if(self.iterations%1000==0): 
                        self.printStats(testSet, testOutput, testSamples)
                        
                gradient /= batchSize
                for i in range(len(self.layerList) - 1):
                    gradient = self.layerList[len(self.layerList) - 1 - i].backPropagation(self.layerList[len(self.layerList)- 2 - i].output, gradient)
                gradient = self.layerList[0].backPropagation(inputMatrix, gradient)
                
            self.batchError /= len(inputSet)
            print("epoch:", epoch, "/",epochs, " -  error:", self.batchError)
            
    
    def testClassifier(self, testSet, testOutput, testSamples=0):
        count = 0
        for i in range(testSamples): 
            j = random.randrange(len(testSet) - 1)
            if(np.argmax(self.guess(testSet[j])) == np.argmax(testOutput[j]).T):
                count += 1
        print(count*100/testSamples, "% " + "over " + str(testSamples) + (" tests"))

    def dumpWeights(self):
        file = open("w/weights" + self.iterations + ".txt", "w")
        
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

    def updateLoss(self, out, pred):
        self.batchError += lossFunctions.mse(out, pred)
        self.itError += lossFunctions.mse(out, pred)
        return lossFunctions.mse_prime(out, pred)
    
    def printStats(self, testSet, testOutput, testSamples):
        self.itError /= 1000
        print("iterations:", self.iterations, "error:", self.itError)
        self.itError = 0
        if(self.mode == "classifier"):
            if(testSet == None):
                testSet = self.inputSet
                testOutput = self.outputSet
            self.testClassifier(testSet, testOutput, testSamples)
        #self.dumpWeights()
        