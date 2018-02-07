#from mohan

from keras.layers import Dense,Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Sequential,Model
from keras.optimizers import adam
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import glob
import cv2
import numpy
import random


#building model
model = Sequential()
model.add(Conv2D(4,(3,3),activation='relu',border_mode='same',input_shape=(28,28,1)))# => (None, 28, 28, 4)
model.add(MaxPooling2D((2,2)))# => (None, 14, 14, 4)
model.add(Conv2D(4,(2,2),activation='relu',border_mode='same'))# => (None, 14, 14, 4)
model.add(MaxPooling2D((2,2))) # => (None, 7, 7, 4)
model.add(Conv2D(4,(2,2),activation='relu',border_mode='same')) # => (None, 7, 7, 4)
model.add(UpSampling2D((2,2))) # (None, 14, 14, 4)
model.add(Conv2D(4,(2,2),activation='relu',border_mode='same'))# (None, 14, 14, 4)
model.add(UpSampling2D((2,2))) # (None, 28, 28, 4)
model.add(Conv2D(1,(4,4),activation='relu',border_mode='same')) # (None, 28, 28, 1)

#setting up the parameters
model.compile(optimizer='adam', loss='mse')
model.summary()

x_train = []

images = glob.glob('/home/raghu/Downloads/mnist_png/training/'+str(1)+'/*.png')
for name in images:
    img = cv2.imread(name,cv2.IMREAD_GRAYSCALE);
    img = img_to_array(img)

    x_train.append(list(img))
#model.fit(numpy.asarray(x_train),x_train)
model.fit(x_train, x_train, batch_size=64, nb_epoch=10)



x_test=[]
images = glob.glob('/home/raghu/Downloads/mnist_png/testing/'+str(1)+'/*.png')
for name in images:
    img = cv2.imread(name,cv2.IMREAD_GRAYSCALE);
    img = img_to_array(img)
    img = list(img)


    '''
    ADDING NOICE TO IMAGE
    '''
    for i in range(50):
        x = numpy.random.randint(28)
        y = numpy.random.randint(28)
        img[x][y]=150


    x_test.append(img)


#predicting the output image form noisy image
decoded_imgs = model.predict(numpy.asarray(x_test))


n = 5
plt.figure(figsize=(20, 4))
for i in range(n):

    #display real image
    plt.imshow(numpy.asarray(x_test[i]).reshape(28, 28))
    plt.gray()
    plt.show()

    #display predicted image
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    plt.show()
