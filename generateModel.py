import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
import numpy as np
import os
# import dlib


# Data Generator to load datasets with different orientations
image_path=r'dataset/'
train_datagen=ImageDataGenerator(rescale=1.0/255.0,
                                            rotation_range=20,
                                            zoom_range=0.15,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.15,
                                            horizontal_flip=True,
                                            fill_mode="nearest")

labels=[]
data=[] 

# loading the datasets into numpy arrays
for label in os.listdir(image_path):
    for imgPath in os.listdir(os.path.join(image_path,label)):
        image=load_img(os.path.join(os.path.join(image_path,label),imgPath),target_size=(224,224))
        image=img_to_array(image)
        image = preprocess_input(image)
        
        labels.append(label)
        data.append(image)

data=np.array(data,dtype="float32")
labels=np.array(labels)


# one-hot encoding
encoder=LabelBinarizer()
labels=encoder.fit_transform(labels)
labels=to_categorical(labels)

# splitting train and test sets
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=42)

#base model
baseModel=MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

headModel=baseModel.output
headModel=AveragePooling2D((7,7))(headModel)
headModel=Flatten()(headModel)
headModel=Dense(128,activation="relu")(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation="softmax")(headModel)

model=Model(inputs=baseModel.input,outputs=headModel)

# setting the base model layers as untrainable
for layer in baseModel.layers:
    layer.trainable=False


INIT_LR = 1e-4
EPOCHS = 8
BS = 32
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#compliling the moel
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])

#training the model
history=model.fit(train_datagen.flow(trainX,trainY,batch_size=BS),
                 steps_per_epoch=len(trainY)/BS,
                 validation_data=(testX,testY),
                 validation_steps=len(testY)/BS,
                 verbose=1,
                 epochs=EPOCHS)


#saving the model
model.save(os.path.join('model','trained_model.h5'))