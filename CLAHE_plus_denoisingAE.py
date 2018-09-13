from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Dropout, Concatenate, MaxPooling2D, UpSampling2D, Average
from keras.layers import Reshape, Conv2DTranspose, BatchNormalization, Activation, AveragePooling2D
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.datasets import cifar10, mnist
from keras.utils import plot_model
from keras import backend as K
import cv2
import numpy as np
import os
from PIL import Image
import histogram_equalization as hist

"""
written by wooramkang 2018.08.20

referenced from lots of papers and gits

if i write, the list of those will be tons of lines haha
"""

def Autoencoder_residualnet(inputs, input_shape):
    '''
    written by wooramkang 2018.09.12

    including resnet = > quite good?
    '''
#   dropout_rate = 0.4
    kernel_size = 3
    filter_norm = input_shape[1]
    print(input_shape)
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    channels = 3
    x = inputs

    x1 = []
    layer_filters = [int(filter_norm / 4), int(filter_norm/2), int(filter_norm)]

    x = Conv2DTranspose(int(filter_norm/2), kernel_size, strides=1,activation='relu',padding='same')(x)

    for filters in layer_filters:
        x_raw = AveragePooling2D()(x)
        x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)
        x1.append(x)
        x = Concatenate(axis=channel_axis)([x, x_raw])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    count = 2
    for filters in layer_filters[::-1]:
        x = Concatenate(axis=channel_axis)([x, x1[count]])
        x = Conv2DTranspose(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        count= count- 1

    del x1

    '''
    written by wooramkang  2018.09.12
    Max         Avg pool => ?
    
    Densenet    Resnet => ?
    '''
    layer_filters = [int(filter_norm), int(filter_norm / 2), int(filter_norm / 4)]
    x_t = x
#    x = Dropout(dropout_rate)(x)
    x = Concatenate(axis=channel_axis)([x, inputs])

    for filters in layer_filters:
        x_t = Conv2D(filters=filters,
               kernel_size=(3, 1),
               strides=1,
               activation='relu',
               padding='same')(x)
        x_t = BatchNormalization()(x_t)
        x_t = Activation('relu')(x_t)

        x = Conv2D(filters=filters,
               kernel_size=(1, 3),
               strides=1,
               activation='relu',
               padding='same')(x_t)
        x = BatchNormalization()(x)/home/rd/recognition_reaserch/FACE/AutoEncoders_With_CLAHE_for_preprocessing/saved_images/sumof_img_gen.png
        x = Activation('relu')(x)
        x = Concatenate(axis=channel_axis)([x, inputs])


    x = Conv2D(filters=3,
                 kernel_size=(3, 3),
                 strides=1,
                 activation='relu',
                 padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def main_color():
    # load the CIFAR10 data
    (x_train, _), (x_test, y_test) = cifar10.load_data()
    #(x_train, _), (x_test, _) = mnist.load_data()
    """
    for preprocessing,
    RGB to LAB
    
    for img in all of img
        DO CLAHE
    
    LAB to RGB 
    """
    x_train_prime = []
    for _img in x_train:
        #_img = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
        #_img = cv2.resize(_img, (80, 80))
        #_img = hist.preprocessing_hist(_img)
        x_train_prime.append(_img)
    x_train = np.array(x_train_prime)
    print(x_train.shape)

    x_test_prime = []
    for _img in x_test:
        #_img = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
        #_img = cv2.resize(_img, (80, 80))
        #_img = hist.preprocessing_hist(_img)
        x_test_prime.append(_img)
    x_test = np.array(x_test_prime)
    print(x_test.shape)
    """
    written by wooramkang 2018.08.21

    depending on CLAHE parameters,
    
    depending on dataset, you could use resizing and colorizing as well 
    
    08.22
    filter grid size    
        2 * 2
        4 * 4
        8 * 8
        16 * 16
    """

    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channels = x_train.shape[3]

    imgs_dir = 'saved_images'
    save_dir = os.path.join(os.getcwd(), imgs_dir)
    if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    imgs = x_test[:100]
    print(y_test[:100])
    i = 0
    for _img in imgs:
        i = i+1
        Image.fromarray(_img).save('saved_images/{0}_img_raw.png'.format(i))
    #print raw img each image by image

    imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    Image.fromarray(imgs).save('saved_images/sumof_img_raw.png')

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

    input_shape = (img_rows, img_cols, 3)
    print(input_shape[1])
    batch_size = 32
    kernel_size = 3
    latent_dim = 256
    layer_filters = [64, 32, 16]

    inputs = Input(shape=input_shape, name='encoder_input')
    model_dim = Autoencoder_residualnet(inputs, input_shape)
    x = inputs


    for filters in layer_filters:
        x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=1,
               activation='relu',
               padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)

    for filters in layer_filters[::-1]:
        x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=1,
               activation='relu',
               padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)

    x = Dropout(rate=0.2)(x)
    x = Conv2D(filters=3,
                              kernel_size=kernel_size,
                              strides=1,
                              activation='sigmoid',
                              padding='same',
                              name='finaloutput_AE'
                              )(x)

    outputs = x

    #decoder = Model(latent_inputs, outputs, name='decoder')
    #decoder.summary()

    #autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')

    autoencoder = Model(inputs, model_dim, name='autoencoder')
    autoencoder.summary()
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'AE_model.{epoch:03d}.h5'
    if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   verbose=1,
                                   min_lr=0.5e-6)

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)

    autoencoder.compile(loss='mse', optimizer='adam')
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    callbacks = [lr_reducer, checkpoint, early]

    # .fit(data for train, data for groundtruth, validtation set, epochs, batchsize, ...)
    autoencoder.fit(x_train,
                    x_train,
                    validation_data=(x_test, x_test),
                    epochs=10,
                    batch_size=batch_size,
                    callbacks=callbacks)
    autoencoder.summary()

    x_decoded = autoencoder.predict(x_test)

    imgs = x_decoded[:100]
    print(imgs.shape)
    imgs = (imgs * 255).astype(np.uint8)

    i = 0
    for _img in imgs:
        i = i + 1
        Image.fromarray(_img).save('saved_images/{0}_img_gen.png'.format(i))
    #print generated img each image by image

    imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    Image.fromarray(imgs).save('saved_images/sumof_img_gen.png')

if __name__ == "__main__":
    main_color()



