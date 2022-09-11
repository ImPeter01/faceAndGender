import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from matplotlib import cm
import cv2
from skimage.filters import threshold_local
import os
import imutils
import glob
import random

#Initial parameter.
WIGHT = 96
HEIGHT = 96
CHANNEL = 3
class_num = 2
batch_size = 128
epochs = 50
path_gender = "D:/0. HTP/Laptrinh/Gender_detect/Model-Demo/Gender_dataset"

data = []
label = []
count = 0

#Use glob to read data set traning
#glob.glob(pattern, *, recursive=False)
#get all man and woman picture.

img_list = [instance for instance in glob.glob(path_gender + "/**/*", recursive=True) if not os.path.isdir(instance)]
random.shuffle(img_list)

for img in img_list:
  img_read = cv2.imread(img)
  img_read = cv2.resize(img_read,(WIGHT,HEIGHT))
  img_read = img_to_array(img_read)

  data.append(img_read)

  name_label = img.split(os.path.sep)[-2] #/content/drive/MyDrive/Face and gender/gender_dataset_face/woman/face_1162.jpg
  if name_label == "woman":
    name_label = 1
  else:
    name_label = 0

  label.append(name_label)
  count = count +1
  print("processing: ", count)

data = np.array(data,dtype="float")/255.0
label = np.array(label)

print(len(data))
print(len(label))

# create data train with train_test_split
(x_train,x_test,y_train,y_test) = train_test_split(data,label,test_size=0.2,random_state= 42)

# covert y_train, y_test to binary 0 1

y_test = keras.utils.to_categorical(y_test, num_classes= 2) # 2 output 0 va 1
y_train = keras.utils.to_categorical(y_train, num_classes= 2)

#print(x_train.shape)
#print(y_train.shape)

#augmenting datset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

model = build(HEIGHT,WIGHT,CHANNEL,class_num)

#compile model

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",
                                                                          tf.keras.metrics.Precision(),
                                                                          tf.keras.metrics.Recall()])
H = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,steps_per_epoch = len(x_train)//batch_size, validation_data=(x_test,y_test), verbose=1)
#save model

model.save('gender_detect.model')

plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.show()
