#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras.optimizers
import os

img_width, img_height = 224, 224

data_root_dir = '/media/mingrui/DATA/datasets'
#data_root_dir = '/media/brainteam/hdd1/201801-IDH'

train_data_dir = os.path.join(data_root_dir, '201801-IDH-jpeg-train')
validation_data_dir = os.path.join(data_root_dir, '201801-IDH-jpeg-validation')

def train():
    nb_train_samples = 5000
    nb_validation_samples = 700
    epochs = 500
    batch_size = 16
    learning_rate = 0.001

    decay_rate = learning_rate / epochs

    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Dense(5, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        #horizontal_flip = True,
        #vertical_flip = True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
    )

    model.save_weights('first_try.h5')
    model.save('model.h5')

if __name__ == '__main__':
    print('modality binary classification')
    train()
