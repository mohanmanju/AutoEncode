from keras.models import Sequential,model_from_json
from keras.layers import Dense,Activation
from keras.optimizers import adam
from keras.preprocessing.image import img_to_array
import cv2
import numpy

model = Sequential()
model.add(Dense(units=300,activation='relu',input_dim=100*100))
model.add(Dense(units=200,activation='relu'))
model.add(Dense(units=100,activation='relu'))
model.add(Dense(units=50,activation='relu'))
model.add(Dense(units=100,activation='relu'))
model.add(Dense(units=200,activation='relu'))
model.add(Dense(units=640000))

model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

img = cv2.imread('../abc.jpg',cv2.IMREAD_GRAYSCALE)
img_data = img_to_array(img)
img_data = img_data.reshape(1,10000)


img_target = cv2.imread('../xyz.jpg',cv2.IMREAD_GRAYSCALE)
img_data_target = img_to_array(img_target)
img_data_target = img_data_target.reshape(1,640000)

model.fit(img_data,img_data_target,epochs=50)

result = model.predict(numpy.asarray(img_data))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

#resizing the input and output image
image_result = result.reshape(800, 800)
print image_result
cv2.imshow("shdg",img)
cv2.waitKey(0)
cv2.imshow("shdg",img_target)
cv2.waitKey(0)


#view the result
import matplotlib.pyplot as plt
plt.imshow(image_result)
plt.gray()
plt.show()
cv2.destroyAllWindows()