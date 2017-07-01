# Script inspirado e baseado em https://blog.keras.io/building-powerful-image-
#classification-models-using-very-little-data.html

# Classificador de cachorros e gatos

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import cv2 as cv
import numpy as np
import os

# Monta a rede
def build_in():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

img_width, img_height = 150, 150

train_data_dir = 'images/training'
validation_data_dir = 'images/validation'
nb_train_samples = 1000
nb_validation_samples = 400
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
# Verifica se a rede ja foi treinada. Se sim, entao deve haver um arquivo que
# armazena os pesos calculados. Senao, calcula os pesos.
if os.path.isfile('models/first_try.h5'):
    # Se a rede ja foi treinada anteriormente, entao testa:
    for i in range(1, 16): # percorre as 15 imagens de teste
        model = build_in() 
        model.load_weights('models/first_try.h5') # carrega na rede os pesos lidos do arquivo first_try.h5
        img = image.load_img('images/teste%d.jpg' % i, target_size = (150, 150))
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img) # joga a imagem de teste contra a rede
         
        print 'chance de teste%d ser cachorro: %f, de ser gato: %f'\
            % (i, prediction[0][0], 1-prediction[0][0])
else:
    model = build_in() # monta
    model.compile(loss = 'binary_crossentropy',
                optimizer = 'rmsprop',
                metrics = ['accuracy'])

    # Training
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size)

    model.save_weights('models/first_try.h5')
