import random
import numpy as np
import pygame
from layers.convolutionLayer import kernelLayer
import activationFunctions
from layers.layer import layer
import lossFunctions
from layers.maxPoolLayer import maxPoolLayer
from layers.reshapeLayer import reshapeLayer
from layers.denseLayer import dense
from time import time

class NeuralNetwork:
    layerList = []
    printItCount = 1000
    deltas = []
    deltaBias = []
    
    def __init__(self, inputShape, lr=0.2):
        self.inputShape = inputShape
        self.lr = lr
        self.loss = lossFunctions.binary_cross_entropy
        self.lossPrime = lossFunctions.binary_cross_entropy_prime
        self.initializeDeltas()
        
    def reinit(self):
        for i in self.layerList:
            i.reinit()
        
    def getInSize(self):
        prevSize = self.inputShape
        if(len(self.layerList) > 0):
            prevSize = self.layerList[len(self.layerList) - 1].outputShape
        return prevSize
    
    def addDenseLayer(self, size, activationFunction=activationFunctions.leakyRelu, activationFunctionD=activationFunctions.leakyReluD):
        prevSize = self.inputShape[0]
        if(len(self.layerList) > 0):
            prevSize = self.layerList[len(self.layerList) - 1].outputShape[0]
        l = dense((prevSize, 1), (size, 1), activationFunction, activationFunctionD)
        self.layerList.append(l)
        
    def addConvLayer(self, kernelSize, kernelDepth=1, activation=activationFunctions.leakyRelu, activationD=activationFunctions.leakyReluD, mode='valid'):
        l = kernelLayer(self.getInSize(), kernelSize, kernelDepth, activation, activationD, mode=mode)
        self.layerList.append(l)
        
    def addReshapeLayer(self, outputShape):
        l = reshapeLayer(self.getInSize(), outputShape)
        self.layerList.append(l)
        
    def addMaxPoolLayer(self):
        l = maxPoolLayer(self.getInSize())
        self.layerList.append(l)
    
    def addLayer(self, layer):
        self.layerList.append(layer)
        
    def guess(self, input):
        outputMatrix = np.array(input, ndmin=2)
        for it in range(len(self.layerList)):
            outputMatrix = self.layerList[it].forward(outputMatrix)
        return outputMatrix

    def train(self, inputSet, outputSet, epochs, mode="", testSamples=0, testSet=[], testOutput=[], batchSize=1):
        self.startTime = time()
        self.iterations = 0
        self.inputSet = inputSet
        self.outputSet = outputSet
        self.testSet = testSet
        self.testOutput = testOutput
        self.epochs = epochs
        self.mode = mode
        self.testSamples = testSamples
        self.itTime = time()
        self.itError = 0
        self.initializeDeltas()
        self.batchError = 0
        
        for epoch in range(epochs):
            for batch in range(len(inputSet) // batchSize):
                for k in range(batchSize):
                    j = random.randrange(0, len(outputSet))          
                    outputTarget = np.array(outputSet[j], ndmin=2).T
                    inputMatrix = np.array(inputSet[j], ndmin=2)
                    outputMatrix = self.guess(inputMatrix)
                    lossGradient = self.updateLoss(outputTarget, outputMatrix)
                    self.updateGradient(lossGradient, inputMatrix)
                    self.iteration()
                self.updateWeights(self.deltas)
                self.updateBias(self.deltaBias)
                self.initializeDeltas()
                self.batch()
            self.epoch(epoch)
            
    
    def testClassifier(self, testSet, testOutput, testSamples=0):
        count = 0
        for i in range(testSamples): 
            j = random.randrange(len(testSet) - 1)
            if(np.argmax(self.guess(testSet[j])) == np.argmax(testOutput[j]).T):
                count += 1
        print(count*100/testSamples, "% " + "over " + str(testSamples) + (" tests using test set"))
        count = 0
        for i in range(min(testSamples, len(self.inputSet))): 
            j = random.randrange(len(self.inputSet))
            if(np.argmax(self.guess(self.inputSet[j])) == np.argmax(self.outputSet[j]).T):
                count += 1
        print(count*100/testSamples, "% " + "over " + str(testSamples) + (" tests using training set"))

    def dumpWeights(self):
        file = open("w/weights" + str(self.iterations) + ".txt", "w")
        
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
        self.batchError += self.loss(out, pred)
        self.itError += self.loss(out, pred)
        return self.lossPrime(out, pred)
    
    def printStats(self, testSet, testOutput, testSamples):
        print("iterations:", self.iterations, "error:", self.itError/1000)
        if(self.mode == "classifier"):
            if(testSet == []):
                testSet = self.inputSet
                testOutput = self.outputSet
            self.testClassifier(testSet, testOutput, testSamples)
            
    def printStatsEpoch(self, testSet, testOutput, testSamples):
        print("iterations:", self.iterations, "error:", self.itError/self.printItCount)
        if(self.mode == "classifier"):
            if(testSet == None):
                testSet = self.inputSet
                testOutput = self.outputSet
            self.testClassifier(testSet, testOutput, testSamples)

        
    def iteration(self):
        self.iterations += 1
        if(self.iterations % self.printItCount == 0): 
            self.printStats(self.testSet, self.testOutput, self.testSamples)
            print("iteration Time: ", time() - self.itTime)
            self.itError = 0
            self.itTime = time()
            
    def updateWeights(self, gradient):
        for i in range(len(self.layerList)):
            self.layerList[i].updateWeights(gradient[i])
        
            
    def updateBias(self, gradient):
        for i in range(len(self.layerList)):
            self.layerList[i].updateBias(gradient[i])
                    
    def updateGradient(self, gradient, inputMatrix):
        layerCount = len(self.layerList)
        for i in range(layerCount-1):
            self.deltas[layerCount - 1  - i] += self.layerList[len(self.layerList) - 1 - i].deltaWeights(self.layerList[layerCount - 2 - i].output, gradient)
            if(self.deltaBias[layerCount - 1  - i].size != 0):
                self.deltaBias[layerCount - 1  - i] += gradient
            gradient = self.layerList[layerCount - 1 - i].backPropagation(self.layerList[layerCount - 1 - i].output, gradient)
        self.deltas[0] += self.layerList[0].deltaWeights(inputMatrix, gradient)
        if(self.deltaBias[0].size != 0):
            self.deltaBias[0] += gradient
        
    def epoch(self, epoch):
        self.batchError /= len(self.inputSet)
        print("epoch:", epoch, "/", self.epochs, " -  error:", self.batchError)
        self.batchError = 0
        
    def batch(self):
        pass
    
    def initializeDeltas(self):
        self.deltas.clear()
        self.deltaBias.clear()
        for i in self.layerList:
            self.deltas.append(np.zeros_like(i.weights))
            self.deltaBias.append(np.zeros_like(i.bias, dtype=float))