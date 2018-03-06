#from mohan

from keras.layers import Dense,Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Sequential,Model
from keras.optimizers import adam
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
import matplotlib.pyplot as plt
import glob
import cv2
import numpy
import random
#building model
model = Sequential()                                                        #height,width
model.add(Conv2D(100,(2,2),activation='relu',border_mode='same',input_shape=(100,100,1)))# => (None, 28, 28, 4)
model.add(MaxPooling2D((2,2)))# => (None, 14, 14, 4)
model.add(Conv2D(200,(2,2),activation='relu',border_mode='same'))# => (None, 14, 14, 4)
model.add(MaxPooling2D((2,2))) # => (None, 7, 7, 4)
model.add(Conv2D(50,(2,2),activation='relu',border_mode='same')) # => (None, 7, 7, 4)
model.add(UpSampling2D((2,2))) # (None, 14, 14, 4)
model.add(Conv2D(100,(2,2),activation='relu',border_mode='same'))# (None, 14, 14, 4)
model.add(UpSampling2D((2,2))) # (None, 28, 28, 4)
model.add(Conv2D(200,(2,2),activation='relu',border_mode='same')) # (None, 28, 28, 1)
model.add(UpSampling2D((2,2))) # (None, 28, 28, 4)
model.add(Conv2D(100,(2,2),activation='relu',border_mode='same')) # (None, 28, 28, 1)
model.add(UpSampling2D((2,2))) # (None, 28, 28, 4)
model.add(Conv2D(50,(2,2),activation='relu',border_mode='same')) # (None, 28, 28, 1)
model.add(UpSampling2D((2,2))) # (None, 28, 28, 4)
model.add(Conv2D(1,(2,2),activation='relu',border_mode='same')) # (None, 28, 28, 1)
# model.add(UpSampling2D((2,2))) # (None, 28, 28, 4)
# model.add(Conv2D(3,(2,2),activation='relu',border_mode='same')) # (None, 28, 28, 1)



#setting up the parameters
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
model.summary()


#data the data
x_train = []
x_target = []
#images = glob.glob('/home/raghu/Desktop/mohan/finalYear/vehicle/objects/*.jpg')
images = ['abc.jpg']
for name in images:
    img_main = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    img_main_array = img_to_array(img_main)
    img_target = cv2.imread('xyz.jpg', cv2.IMREAD_GRAYSCALE)
    img_target_array = img_to_array(img_target)
    x_train.append(list(img_main_array))
    x_target.append(list(img_target_array))
    break




#model.fit(numpy.asarray(x_train),x_train)
model.fit(numpy.asarray(x_train), numpy.asarray(x_target), batch_size=64, nb_epoch=5000)
result = model.predict(numpy.asarray(x_train))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

#resizing the input and output image
image_result = result.reshape(800, 800)
print image_result
cv2.imshow("shdg",img_main)
cv2.waitKey(0)
cv2.imshow("shdg",img_target)
cv2.waitKey(0)


#view the result
import matplotlib.pyplot as plt
plt.imshow(image_result)
plt.gray()
plt.show()
cv2.destroyAllWindows()

