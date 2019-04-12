from keras import backend as K
from keras import layers
from keras.layers import Conv2D,Activation,MaxPooling2D,BatchNormalization,Conv2DTranspose,UpSampling2D,Concatenate,Input
from keras import regularizers
from keras.models import Model,Sequential

import matplotlib.pyplot as plt
import numpy as np

# import keras
# print("\n The keras version is ",keras.__version__)
# print("\n The TF version is ",tf.__version__)
K.set_learning_phase(1) #set learning phase
K.set_image_data_format('channels_last')


L2R = 1e-6
reg = regularizers.l2(L2R)
def stackedCNN(x,filters,level,namepre): # two layers of stacked CNN+BN+ReLU
    num_convs = 2
    SCNet = Sequential(name=namepre+'_SC_' + str(level))
    for i in range(num_convs):
        SCNet.add(Conv2D(filters=filters, kernel_size=(3,3), padding='same',kernel_regularizer=reg, bias_regularizer=reg))
        SCNet.add(BatchNormalization())
        SCNet.add(Activation('relu'))

    x = SCNet(x)
    return x

def downsampling(x, level, filters):
    x = stackedCNN(x,filters,level,'downsampling')
    skip = x
    x = MaxPooling2D(pool_size=(2,4))(x)
    return x, skip


def bottleneck_dilated(x, filters):
    skips = []
    for i in range(2):
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=1, dilation_rate=(2 ** i,2 ** i), activation='relu',
                   padding='same', kernel_regularizer=reg, bias_regularizer=reg, name='bottleneck_skip_' + str(i))(x)
        skips.append(x)
    x = layers.add(skips)
    return x





def upsampling(x, level, skip, filters):
    # x1 = Conv2DTranspose(filters=filters, kernel_size=(3,5),strides=(2,4),
    #                     name='upsampling_' + str(level) + '_conv_trans_' + str(level))(x)
    x = UpSampling2D((2,4), name='upsampling_' + str(level))(x)
    x = Concatenate()([x, skip])

    x = stackedCNN(x, filters, level, 'upsampling')

    return x


def commonProcess(Nsrc, num_channels):

    num_freq = 512
    num_frame = 64
    img_shape = (num_frame, num_freq, num_channels)

    dim_output = 20


    num_levels = 3
    filter_size_start = 32

    inputs = Input(img_shape)
    x = inputs
    skips = []
    for i in range(num_levels):
        x, skip = downsampling(x, i, filter_size_start * (2 ** i))
        skips.append(skip)

    x = bottleneck_dilated(x, filter_size_start * (2 ** num_levels))

    for j in range(num_levels):
        x = upsampling(x, j, skips[num_levels - j - 1], filter_size_start * (2 ** (num_levels - j - 1)))

    split = Conv2D(filters=dim_output, kernel_size=(3,3), padding='same', activation='linear',
                   name='split',kernel_regularizer=reg, bias_regularizer=reg)(x)

    ################################################################## one head
    ZRA = layers.Activation('tanh')(split)
    def DCoutput(inp):
        from keras import backend as K
        y = K.l2_normalize(inp, axis=-1)
        return y

    V = layers.Lambda(DCoutput,name='embedding')(ZRA)

    ################################################################## another head
    def reshapeEasy(inp, target_shape):
        from keras import layers
        inputR = layers.Reshape(target_shape=target_shape)(inp)
        return inputR

    ZR2 = layers.Lambda(reshapeEasy, arguments={'target_shape':(num_frame*num_freq,dim_output)})(split)
    MR = layers.TimeDistributed(layers.Dense(Nsrc,activation='softmax'))(ZR2)
    M = layers.Lambda(reshapeEasy,arguments={'target_shape':(num_frame,num_freq,Nsrc)},name='mask')(MR)

    BSS_model = Model(inputs=[inputs], outputs=[V, M])

    return BSS_model

def ChimeraNet(Nsrc = 2, inputFeature=0):
    # inputFeature 0: spectral+ numSrc shifted DOA spatial features
    #              1: spectral + DOA spatial features
    #              2: spectral features
    channels = [Nsrc+1,2,1][inputFeature]
    BSS_model = commonProcess(Nsrc, channels)
    return BSS_model


if __name__=="__main__":
    train_model = ChimeraNet(Nsrc=2, inputFeature=0)
    print(train_model.summary())
    from keras.utils import plot_model
    plot_model(train_model, to_file='Model.png')

    aa = 0

