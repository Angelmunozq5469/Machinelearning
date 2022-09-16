##I changed "tensorflow.keras.preprocessing.image" to "keras_preprocessing.image" and solved the issue.


from ctypes import resize
import io
from tkinter import Image, image_names
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import os
import numpy as np


#img=image.load_img("TALLER 5 (DEFINIR)/DATASET/train/NORMAL/IM-0115-0001.jpeg")
#plt.imshow(img)
#plt.waitforbuttonpress(0)
#img2=cv.imread("TALLER 5 (DEFINIR)/DATASET/train/NORMAL/IM-0115-0001.jpeg").shape
#print (img2)

entrenar=ImageDataGenerator(rescale=1/255)
validacion=ImageDataGenerator(rescale=1/255)

#entreno con mi data train

entrenar_data=entrenar.flow_from_directory("TALLER 5 (DEFINIR)/DATASET/train/",
                                        target_size=(255,255),
                                        batch_size=20,
                                        class_mode="binary")

validacion_data=validacion.flow_from_directory("TALLER 5 (DEFINIR)/DATASET/validation/",
                                        target_size=(255,255),
                                        batch_size=2,
                                        class_mode="binary")
#NORMAL:0 AND PNEUMONIA:1
indices=entrenar_data.class_indices
#ARRAY CON RESULTADOS
array=entrenar_data.classes

model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape=(255,255,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512,activation="relu"),
                                    tf.keras.layers.Dense(1,activation="sigmoid")
                                    ])

model.compile(loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
            metrics=['accuracy'])




model_fit=model.fit(entrenar_data,epochs=1,batch_size=10)
dir_path="TALLER 5 (DEFINIR)/DATASET/test/"



for i in os.listdir(dir_path):

    img=image.load_img(dir_path + '//' +i,target_size=(255,255)) #ERROR
    plt.imshow(img)
    plt.show()
    
    X=image.img_to_array(img)
    X=np.expand_dims(X,axis=0)
    images=np.vstack([X])
    val=model.predict(images)
    if val==0:
        print("neumonia")
    else:
            print("normales")
    
    