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
model = Sequential()                                                        #height,width
model.add(Conv2D(10,(2,2),activation='relu',border_mode='same',input_shape=(40,40,3)))# => (None, 28, 28, 4)
model.add(MaxPooling2D((2,2)))# => (None, 14, 14, 4)
model.add(Conv2D(20,(2,2),activation='relu',border_mode='same'))# => (None, 14, 14, 4)
model.add(MaxPooling2D((2,2))) # => (None, 7, 7, 4)
model.add(Conv2D(30,(2,2),activation='relu',border_mode='same')) # => (None, 7, 7, 4)
model.add(UpSampling2D((2,2))) # (None, 14, 14, 4)
model.add(Conv2D(30,(2,2),activation='relu',border_mode='same'))# (None, 14, 14, 4)
model.add(UpSampling2D((2,2))) # (None, 28, 28, 4)
model.add(Conv2D(3,(2,2),activation='relu',border_mode='same')) # (None, 28, 28, 1)
# model.add(UpSampling2D((2,2))) # (None, 28, 28, 4)
# model.add(Conv2D(3,(2,2),activation='relu',border_mode='same')) # (None, 28, 28, 1)
# model.add(UpSampling2D((2,2))) # (None, 28, 28, 4)
# model.add(Conv2D(30,(2,2),activation='relu',border_mode='same')) # (None, 28, 28, 1)
# model.add(UpSampling2D((2,2))) # (None, 28, 28, 4)
# model.add(Conv2D(3,(2,2),activation='relu',border_mode='same')) # (None, 28, 28, 1)
# model.add(UpSampling2D((2,2))) # (None, 28, 28, 4)
# model.add(Conv2D(3,(2,2),activation='relu',border_mode='same')) # (None, 28, 28, 1)

#setting up the parameters
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
model.summary()

x_train = []
x_target = []
images = glob.glob('/home/raghu/Desktop/mohan/finalYear/vehicle/objects/*.jpg')
for name in images:
    img_main = cv2.imread(name)#,cv2.IMREAD_GRAYSCALE)
    #cv2.imshow("lalala", img)
    #k = cv2.waitKey(0)
    img = cv2.resize(img_main,(40,40))
    img_target = cv2.resize(img_main,(40,40))
    img = img_to_array(img)

    x_train.append(list(img))
    x_target.append(list(img_target))
    break
#model.fit(numpy.asarray(x_train),x_train)
model.fit(numpy.asarray(x_train), numpy.asarray(x_target), batch_size=64, nb_epoch=100000)
result = model.predict(numpy.asarray(x_train))
vis2 = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
vis2 = cv2.resize(vis2,(40, 40))
cv2.imshow("shdg",cv2.resize(img_main,(40,40)))
cv2.waitKey(0)
cv2.imshow("shdg",img_target)
cv2.waitKey(0)
cv2.imshow("shdg",vis2)
cv2.waitKey(0)
'''img1 = cv2.blur(img_main,(20,20))
img1 = cv2.resize(img1,(400,400))
cv2.imshow("shdg",img1)
cv2.waitKey(0)
img2 = cv2.resize(img_main,(800,800))
cv2.imshow("shdg",img2)
cv2.waitKey(0)'''
#
#
# x_test=[]
# images = glob.glob('/home/raghu/Downloads/mnist_png/testing/'+str(1)+'/*.png')
# for name in images:
#     img = cv2.imread(name,cv2.IMREAD_GRAYSCALE);
#     cv2.imshow("lalala", image)
#     k = cv2.waitKey(0)
#     break
#     img = img_to_array(img)
#     img = list(img)
#
#
#     '''
#     ADDING NOICE TO IMAGE
#     '''
#     for i in range(50):
#         x = numpy.random.randint(28)
#         y = numpy.random.randint(28)
#         img[x][y]=150
#
#
#     x_test.append(img)
#
#
# #predicting the output image form noisy image
# decoded_imgs = model.predict(numpy.asarray(x_test))
#
#
# n = 5
# plt.figure(figsize=(20, 4))
# for i in range(n):
#
#     #display real image
#     plt.imshow(numpy.asarray(x_test[i]).reshape(28, 28))
#     plt.gray()
#     plt.show()
#
#     #display predicted image
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     plt.show()
