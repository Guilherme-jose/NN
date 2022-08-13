from random import randrange
from PIL import Image
import numpy as np
import neuralNetwork as nn
import activationFunctions
from lossFunctions import mse
from lossFunctions import mse_prime
import matplotlib.pyplot as plt

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


nn = nn.NeuralNetwork((784,1))
nn.addReshapeLayer((784,1))
nn.addDenseLayer(256, activationFunctions.sigmoid, activationFunctions.sigmoidD)
nn.addDenseLayer(10, activationFunctions.sigmoid, activationFunctions.sigmoidD)

nn.loss = mse
nn.lossPrime = mse_prime

history = nn.train(trainingSet, trainingOutput, 10, "classifier",1000, batchSize=16)

plt.plot(history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('iterations/1000')
plt.show()

plt.plot(history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('iterations/1000')
plt.show()

print("finished")