from random import randrange
from PIL import Image
import numpy as np
import neuralNetwork as nn
import activationFunctions

def by255(x):
    return x/255
        

trainingSet = []
trainingOutput = []

dataset = open("mnist_train_60000.txt")
imageRes = [28,28]

for i in range(60):
    input = dataset.readline()
    inputArray = input.split()
    label = inputArray.pop()
    inputArray = [by255(int(i)) for i in inputArray]
    inputArray = np.array(inputArray).T
    trainingSet.append(inputArray)
    outputArray = [0,0,0,0,0,0,0,0,0,0] 
    outputArray[int(label)] = 1
    trainingOutput.append(outputArray)
dataset.close()

nn = nn.NeuralNetwork((784,1))
nn.addReshapeLayer(784, (784,1))
nn.addDenseLayer(128, activationFunctions.sigmoid, activationFunctions.sigmoidD)
nn.addDenseLayer(10, activationFunctions.sigmoid, activationFunctions.sigmoidD)
nn.train(trainingSet, trainingOutput, 300, "classifier",600)

print("finished")

