from keras import models
import numpy as np
from PIL import Image
from keras import backend as K

def by255(x):
    return x/255

while(1):
    testSet = []
    shape = (3,64,64)
    print('image path')
    image = Image.open(input()).resize((64,64))
    image.show()
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, shape)
    data = [by255(i) for i in data]
    testSet.append(data)

    testSet = np.array(testSet, ndmin=4)

    model = models.load_model('modelT1')

    print (model.predict(testSet))

