from time import time
import matplotlib.pyplot as plt
import numpy
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.regularizers import L2
import numpy as np
from matplotlib import pyplot
from PIL import Image
from keras import backend as K

def by255(x):
    return x/255

trainingSet = []
trainingOutput = []
testSet = []
testOutput = []

setSize = 3000
testSize = 1000

shape = (3,64,64)
for i in range(1,setSize):
    image = Image.open('training_set_small/cats/cat.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, shape)
    data = [by255(i) for i in data]
    trainingSet.append(data)
    trainingOutput.append([1,0])
    
    image = Image.open('training_set_small/dogs/dog.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, shape)
    trainingSet.append(data)
    trainingOutput.append([0,1])
    
for i in range(setSize, setSize + testSize):
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

trainingSet = np.array(trainingSet, ndmin=4)
trainingOutput = np.array(trainingOutput, ndmin=2)
testSet = np.array(testSet, ndmin=4)
testOutput = np.array(testOutput, ndmin=2)
# Criando o modelo
model = Sequential()
model.add(Conv2D(filters=32, kernel_size= (5, 5), input_shape=(3, 64, 64), padding='same', activation='relu'))
#model.add(Conv2D(filters=72, kernel_size= (3, 3), input_shape=(52, 32, 32), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(256,kernel_regularizer=L2(), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compilando o modelo
epocas = 10
lrate = 0.01
sgd = SGD(learning_rate=lrate)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy' ]) 
print(model.summary())

print("training")
# Treinando o modelo
history = model.fit(trainingSet, trainingOutput, validation_data=(testSet, testOutput), epochs=epocas, batch_size=32)

# Avaliacao final do modelo
scores = model.evaluate(testSet, testOutput, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Sumariza para a precisão
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Sumariza para a mostrar a perda
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('modelT1')
print("finished")