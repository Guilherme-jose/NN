from keras import models
import numpy as np
from PIL import Image
from keras import backend as K

def by255(x):
    return x/255

testSet = []
testOutput = []


shape = (3,64,64)
for i in range(4000,4500):
    image = Image.open('training_set_small/cats/cat.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, shape)
    data = [by255(i) for i in data]
    testSet.append(data)
    testOutput.append([1,0])
    
    image = Image.open('training_set_small/dogs/dog.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, shape)
    testSet.append(data)
    testOutput.append([0,1])

testSet = np.array(testSet, ndmin=4)
testOutput = np.array(testOutput, ndmin=2)

model = models.load_model('modelT1')

model.evaluate(testSet, testOutput, batch_size = 400)

