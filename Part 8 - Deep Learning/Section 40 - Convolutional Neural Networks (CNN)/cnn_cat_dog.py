# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:24:47 2020

@author: st
dataset is not incsv files.
it consits of only jpeg images.
naming conventions -cat.1,cat.2 etc..,

to import dataset from keras-> special datastructure when using the keras model
increase accuracy -> convolution layer 2D add 
increase target size and try in GPUtarget_size=(64, 64)
"""
#Part 1 building CNN.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D # we use for convolusional step
from tensorflow.keras.layers import MaxPooling2D # used for pooling
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
#intialising CNN
classifier=Sequential()
#adding convolusional layer #
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#pooling step
#reduce computing time 
classifier.add(MaxPooling2D(pool_size=(2,2)))

#to improve accuracy level adding 2nd convolution lyer
#input_shape can be removed , bcoz we have previous layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
#pooling step
#reduce computing time 
classifier.add(MaxPooling2D(pool_size=(2,2)))


#flattening convert pool feature maps -> get a single vector
#we won't directly use flattening -> bcoz we won't know the spacial relation between the pixels , hence to know the relation between pixels we undergo convoluton,pooling steps before flattening.
classifier.add(Flatten())
#  full connection step or hidden layer -> classic ANN 
classifier.add(Dense(128,activation = 'relu'))
#add output layer
classifier.add(Dense(1,activation = 'sigmoid'))

#compile whole models
#categorical_crossentropy
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fit the model -> pre processing steps
#keras doc-> imgae augumentation-> prevent over fitting

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',#changes
        target_size=(64, 64),#changes
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set,
        steps_per_epoch=8000, #number of images in training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

'''Epoch 25/25
8000/8000 [==============================] - 1412s 177ms/step - loss: 0.0285 - accuracy: 0.9904 - val_loss: 1.8548 - val_accuracy: 0.7526
'''