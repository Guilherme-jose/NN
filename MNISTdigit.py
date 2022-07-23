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

for i in range(60000):
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


""""
nn = nn.NeuralNetwork((784,1))
nn.addReshapeLayer(784, (784,1))
nn.addDenseLayer(128, activationFunctions.sigmoid, activationFunctions.sigmoidD)
nn.addDenseLayer(10, activationFunctions.sigmoid, activationFunctions.sigmoidD)

"""

nn = nn.NeuralNetwork((784,1))
nn.addReshapeLayer((784,1), (784,1))
#nn.addReshapeLayer((784,1), (1,28,28))
#nn.addConvLayer((1,28,28), 5, 6)
#nn.addMaxPoolLayer((6,24,24), (6,12,12))
#nn.addConvLayer((6,12,12), 5, 16)
#nn.addMaxPoolLayer((16,8,8), (16,4,4))
#nn.addReshapeLayer((16,4,4), (256,1))
nn.addDenseLayer(128, activationFunctions.sigmoid, activationFunctions.sigmoidD)
nn.addDenseLayer(10, activationFunctions.sigmoid, activationFunctions.sigmoidD)


nn.train(trainingSet, trainingOutput, 300, "classifier",100)

print("finished")

