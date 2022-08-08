from random import randrange
from PIL import Image
import numpy as np
import neuralNetwork as nn
import activationFunctions

def by255(x):
    return x/255
        
print("initializing weights")
nn = nn.NeuralNetwork((3,64,64))
nn.addConvLayer(5, 32, mode='same')
nn.addMaxPoolLayer()
nn.addConvLayer(5, 52, mode='same')
nn.addMaxPoolLayer()
nn.addReshapeLayer((13312,1))
nn.addDenseLayer(256, activationFunctions.leakyRelu, activationFunctions.leakyReluD)
nn.addDenseLayer(64, activationFunctions.leakyRelu, activationFunctions.leakyReluD)
nn.addDenseLayer(2, activationFunctions.sigmoid, activationFunctions.sigmoidD)

trainingSet = []
trainingOutput = []
testSet = []
testOutput = []

setSize = 300
testSize = 100

shape = (3,64,64)
for i in range(1,setSize):
    image = Image.open('training_set_small/cats/cat.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, shape)
    data = [by255(i) for i in data]
    trainingSet.append(data)
    trainingOutput.append([1,1e-8])
    
    image = Image.open('training_set_small/dogs/dog.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, shape)
    trainingSet.append(data)
    trainingOutput.append([1e-8,1])
    
for i in range(setSize, setSize + testSize):
    image = Image.open('training_set_small/cats/cat.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, shape)
    data = [by255(i) for i in data]
    testSet.append(data)
    testOutput.append([1,1e-8])
    
    image = Image.open('training_set_small/dogs/dog.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, shape)
    testSet.append(data)
    testOutput.append([1e-8,1])
    
print("training")

nn.train(trainingSet,trainingOutput,1, "classifier", 100, testSet, testOutput, 8)
    
print("finished")

