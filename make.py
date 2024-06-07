import tensorflow as tf 
import matplotlib.pyplot as ma
import numpy as nu
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

import cv2



#for rescaling the image 
train=ImageDataGenerator( rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation = ImageDataGenerator( rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_dataset = train.flow_from_directory('training',target_size=(200,200),batch_size=10,class_mode='sparse')
validation_dataset = validation.flow_from_directory('validation',target_size=(200,200),batch_size=10,class_mode='sparse')
print(len(train_dataset))
print(len(validation_dataset))

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(5,5),activation='relu',input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                  

                                    ##
                                    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                   
                                   
                                    ##
                                    tf.keras.layers.Conv2D(64,(5,5),activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                   

                                    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    
                                  
                                    ##
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dropout(0.5),
                                    ##
                                    tf.keras.layers.Dense(512,activation='relu'),
                                    ##
                                    tf.keras.layers.Dense(40,activation='softmax')
                                    ])
model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])

model_fit =model.fit(train_dataset,steps_per_epoch=len(train_dataset),epochs=200,validation_data=validation_dataset)

model_path = "model.h5"
model.save(model_path)
print("Model saved successfully.")





