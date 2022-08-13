from random import randrange
from PIL import Image
import numpy as np
import neuralNetwork as nn
import activationFunctions
import matplotlib.pyplot as plt

def by255(x):
    return x/255
        
print("initializing weights")
nn = nn.NeuralNetwork((3,64,64))
nn.addMaxPoolLayer()
nn.addConvLayer(3, 32, mode='same')
nn.addMaxPoolLayer()
nn.addConvLayer(3, 52, mode='same')
nn.addMaxPoolLayer()
nn.addReshapeLayer((3328,1))
nn.addDenseLayer(256, activationFunctions.leakyRelu, activationFunctions.leakyReluD)
nn.addDenseLayer(2, activationFunctions.sigmoid, activationFunctions.sigmoidD)

trainingSet = []
trainingOutput = []
testSet = []
testOutput = []

setSize = 1
testSize = 1

shape = (3,64,64)
for i in range(1,setSize+1):
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
    
for i in range(setSize+1, setSize + testSize+1):
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
nn.printItCount = 1
history = nn.train(trainingSet,trainingOutput,1000, "classifier", 1, testSet, testOutput, 1)
    
plt.plot(history['accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('iterations/1000')
plt.show()

print("finished")

