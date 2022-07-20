from random import randrange
from PIL import Image
import numpy as np
import neuralNetwork as nn
import activationFunctions

def by255(x):
    return x/255
        
print("initializing weights")
nn = nn.NeuralNetwork(5250)
nn.addReshapeLayer((5250,1), (1,75,70))
nn.addConvLayer((1,75,70), 7, 6)
nn.addMaxPoolLayer((6,69,64), (6,35,32))
nn.addConvLayer((6,35,32), 5, 16)
nn.addMaxPoolLayer((16,31,28), (16,16,14))
nn.addConvLayer((16,16,14), 5, 26)
nn.addMaxPoolLayer((26,12,10), (26,6,5))
nn.addReshapeLayer((26,6,5), (780,1))
nn.addDenseLayer(512, activationFunctions.sigmoid, activationFunctions.sigmoidD)
nn.addDenseLayer(256, activationFunctions.sigmoid, activationFunctions.sigmoidD)
nn.addDenseLayer(2, activationFunctions.sigmoid, activationFunctions.sigmoidD)

trainingSet = []
trainingOutput = []
for i in range(1,4001):
    image = Image.open('training_set_small/cats/cat.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, 5250)
    data = [by255(i) for i in data]
    trainingSet.append(data)
    trainingOutput.append([1,0])
    
    image = Image.open('training_set_small/dogs/dog.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, 5250)
    data = [by255(i) for i in data]
    trainingSet.append(data)
    trainingOutput.append([0,1])
    
print("training")

nn.train(trainingSet,trainingOutput,10, "classifier")
    
print("finished")

