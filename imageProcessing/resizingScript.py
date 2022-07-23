from PIL import Image
import numpy as np
size = []
print("insert size: X Y")
size.append(input())
size.append(input())
for i in range(4000):
    load_img_rz = np.array(Image.open('training_set/cats/cat.'+ str(i+1) +'.jpg').resize((int(size[1]),int(size[0]))).convert('L'))
    Image.fromarray(load_img_rz).save('training_set_small/cats/cat.'+ str(i+1) +'.jpg')
    print("After resizing:",load_img_rz.shape)
    
for i in range(4000):
    load_img_rz = np.array(Image.open('training_set/dogs/dog.'+ str(i+1) +'.jpg').resize((int(size[1]),int(size[0]))).convert('L'))
    Image.fromarray(load_img_rz).save('training_set_small/dogs/dog.'+ str(i+1) +'.jpg')
    print("After resizing:",load_img_rz.shape)

for i in range(4000, 5000):
    load_img_rz = np.array(Image.open('valset/cats/cat.'+ str(i+1) +'.jpg').resize((int(size[1]),int(size[0]))).convert('L'))
    Image.fromarray(load_img_rz).save('training_set_small/cats/cat.'+ str(i+1) +'.jpg')
    print("After resizing:",load_img_rz.shape)
    
for i in range(4000, 5000):
    load_img_rz = np.array(Image.open('valset/dogs/dog.'+ str(i+1) +'.jpg').resize((int(size[1]),int(size[0]))).convert('L'))
    Image.fromarray(load_img_rz).save('training_set_small/dogs/dog.'+ str(i+1) +'.jpg')
    print("After resizing:",load_img_rz.shape)