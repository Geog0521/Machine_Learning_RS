# -*- coding: utf-8 -*-
"""
@author: Xinghua Cheng, Dept. of Land Surveying and Geo-Informatics, The Hong Kong Polytechnic Univ.
Email: cxh9791156936@gmail.com
"""
import keras
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Activation, BatchNormalization
from keras.layers import BatchNormalization as BN
from keras.initializers import he_normal, RandomNormal
from keras.layers import Dense, Flatten, Add, AveragePooling2D
from keras.layers import GlobalAveragePooling2D,multiply,concatenate

def resnet99_avg(band, imx, ncla1, l=1):
    input1 = Input(shape=(imx,imx,band))

    # define network
    conv0x = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv0 = Conv2D(32,kernel_size=(3,3),padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn11 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv11 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn21 = BatchNormalization(axis=-1,momentum=0.9,epsilon=0.001,center=True,scale=True,
                             beta_initializer='zeros',gamma_initializer='ones',
                             moving_mean_initializer='zeros',
                             moving_variance_initializer='ones')
    conv21 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv22 = Conv2D(64,kernel_size=(3,3),padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    
    fc1 = Dense(ncla1,activation='softmax',name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    # x1
    x1 = conv0(input1)
    x1x = conv0x(input1)
#    x1 = MaxPooling2D(pool_size=(2,2))(x1)
#    x1x = MaxPooling2D(pool_size=(2,2))(x1x)
    x1 = concatenate([x1,x1x],axis=-1)
    x11 = bn11(x1)
    x11 = Activation('relu')(x11)
    x11 = conv11(x11)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    x1 = Add()([x1,x11])
    
    if l==2:
        x11 = bn21(x1)
        x11 = Activation('relu')(x11)
        x11 = conv21(x11)
        x11 = Activation('relu')(x11)
        x11 = conv22(x11)
        x1 = Add()([x1,x11])
    
    x1 = GlobalAveragePooling2D()(x1)
    
#    x1 = Flatten()(x1)
    pre1 = fc1(x1)

    model1 = Model(inputs=input1, outputs=pre1)
    return model1

def seresnet_avg(band, ncla, xke=16, res=2, inx=32):
    inputs = Input(shape=(inx,inx,band))
    x1 = Conv2D(xke*2,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x2 = Conv2D(xke,kernel_size=(5,5),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x3 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)

    x = keras.layers.concatenate([x1,x2,x3],axis=3)

    # residual 1
    for i in range(res):
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x2 = GlobalAveragePooling2D()(x1)
        x2 = Dense(xke//4,
                   kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
        x2 = Dense(xke*4,activation='sigmoid',
                    kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
        x1 = multiply([x1,x2])
        x = Add()([x1,x])
        
    x = AveragePooling2D(2,2)(x)
    x = Conv2D(xke*8,kernel_size=(1,1),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    
    for i in range(res):
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*8,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*8,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x2 = GlobalAveragePooling2D()(x1)
        x2 = Dense(xke//2,
                   kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
        x2 = Dense(xke*8,activation='sigmoid',
                    kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
        x1 = multiply([x1,x2])
        x = Add()([x1,x])
    
    x = AveragePooling2D(2,2)(x)
    x = Conv2D(xke*16,kernel_size=(1,1),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    
    for i in range(res):
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*16,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*16,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x2 = GlobalAveragePooling2D()(x1)
        x2 = Dense(xke,
                   kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
        x2 = Dense(xke*16,activation='sigmoid',
                    kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
        x1 = multiply([x1,x2])
        x = Add()([x1,x])
    
    x = GlobalAveragePooling2D()(x)

    pre = Dense(ncla,activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    model = Model(inputs=inputs, outputs=pre)
    return model

def GermanNet_avg(band,ncla,inx=32):
    inputs = Input(shape=(inx,inx,band))
    x1 = Conv2D(16,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(2,2)(x1)
    x1 = Conv2D(32,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(2,2)(x1)
    x1 = Conv2D(64,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(2,2)(x1)
    x1 = Conv2D(128,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(2,2)(x1)
#    x1 = Flatten()(x1)
    
    x1 = GlobalAveragePooling2D()(x1)
    
    
    x1 = Dense(256,activation='relu')(x1)
    x1 = Dropout(0.5)(x1)

    pre = Dense(ncla,activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    model = Model(inputs=inputs, outputs=pre)
    return model

def GermanNet(band,ncla,inx=32):
    inputs = Input(shape=(inx,inx,band))
    x1 = Conv2D(16,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(2,2)(x1)
    x1 = Conv2D(32,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(2,2)(x1)
    x1 = Conv2D(64,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(2,2)(x1)
    x1 = Conv2D(128,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(2,2)(x1)
    x1 = Flatten()(x1)
    
    
    x1 = Dense(256,activation='relu')(x1)
    x1 = Dropout(0.5)(x1)

    pre = Dense(ncla,activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    model = Model(inputs=inputs, outputs=pre)
    return model



def seresnet_adpt_noSE(band, ncla, xke=16, res=2, inx=32):
    inputs = Input(shape=(inx,inx,band))
    x1 = Conv2D(xke*2,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x2 = Conv2D(xke,kernel_size=(5,5),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x3 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)

    x = keras.layers.concatenate([x1,x2,x3],axis=3)

    # residual 1
    for i in range(res):
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
#        x2 = GlobalAveragePooling2D()(x1)
#        x2 = Dense(xke//4,
#                   kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
#        x2 = Dense(xke*4,activation='sigmoid',
#                    kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
#        x1 = multiply([x1,x2])
        x = Add()([x1,x])
        
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(xke*8,kernel_size=(1,1),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    
    for i in range(res):
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*8,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*8,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
#        x2 = GlobalAveragePooling2D()(x1)
#        x2 = Dense(xke//2,
#                   kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
#        x2 = Dense(xke*8,activation='sigmoid',
#                    kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
#        x1 = multiply([x1,x2])
        x = Add()([x1,x])
    
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(xke*16,kernel_size=(1,1),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    
    for i in range(res):
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*16,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*16,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
#        x2 = GlobalAveragePooling2D()(x1)
#        x2 = Dense(xke,
#                   kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
#        x2 = Dense(xke*16,activation='sigmoid',
#                    kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
#        x1 = multiply([x1,x2])
        x = Add()([x1,x])
    
    x = GlobalAveragePooling2D()(x)

    pre = Dense(ncla,activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    model = Model(inputs=inputs, outputs=pre)
    return model

def seresnet_adpt(band, ncla, xke=16, res=2, inx=32, depth=3):
    inputs = Input(shape=(inx,inx,band))
    x1 = Conv2D(xke*2,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x2 = Conv2D(xke,kernel_size=(5,5),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x3 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)

    x = keras.layers.concatenate([x1,x2,x3],axis=3)

    # residual 1
    if depth>=1:
        for i in range(res):
            x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x)
            x1 = Activation('relu')(x1)
            x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
            x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
            x1 = Activation('relu')(x1)
            x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
            x2 = GlobalAveragePooling2D()(x1)
            x2 = Dense(xke//4,
                       kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
            x2 = Dense(xke*4,activation='sigmoid',
                        kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
            x1 = multiply([x1,x2])
            x = Add()([x1,x])
            
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(xke*8,kernel_size=(1,1),padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    
    # residual 2
    if depth>=2:
        for i in range(res):
            x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x)
            x1 = Activation('relu')(x1)
            x1 = Conv2D(xke*8,kernel_size=(3,3),padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
            x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
            x1 = Activation('relu')(x1)
            x1 = Conv2D(xke*8,kernel_size=(3,3),padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
            x2 = GlobalAveragePooling2D()(x1)
            x2 = Dense(xke//2,
                       kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
            x2 = Dense(xke*8,activation='sigmoid',
                        kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
            x1 = multiply([x1,x2])
            x = Add()([x1,x])
        
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(xke*16,kernel_size=(1,1),padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    
    if depth>=3:
        for i in range(res):
            x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x)
            x1 = Activation('relu')(x1)
            x1 = Conv2D(xke*16,kernel_size=(3,3),padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
            x1 = BN(axis=-1,momentum=0.9,epsilon=0.001)(x1)
            x1 = Activation('relu')(x1)
            x1 = Conv2D(xke*16,kernel_size=(3,3),padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
            x2 = GlobalAveragePooling2D()(x1)
            x2 = Dense(xke,
                       kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
            x2 = Dense(xke*16,activation='sigmoid',
                        kernel_initializer=RandomNormal(mean=0.0,stddev=0.01))(x2)
            x1 = multiply([x1,x2])
            x = Add()([x1,x])
    
    x = GlobalAveragePooling2D()(x)

    pre = Dense(ncla,activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    model = Model(inputs=inputs, outputs=pre)
    return model

def resnet_adpt2(band, ncla, xke=16, res=2, inx=32):
    inputs = Input(shape=(inx,inx,band))
    x1 = Conv2D(xke*2,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x2 = Conv2D(xke,kernel_size=(5,5),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x3 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)

    x = keras.layers.concatenate([x1,x2,x3],axis=3)

    # residual 1
    for i in range(res):
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1,x])
        
    x = MaxPooling2D(2,2)(x)
    
    for i in range(res):
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1,x])
    
    x = MaxPooling2D(2,2)(x)
    
    for i in range(res):
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1,x])
    
#    x = AveragePooling2D(8,8)(x)
    x = Flatten()(x)
    pre = Dense(ncla,activation='softmax',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    model = Model(inputs=inputs, outputs=pre)
    return model

def wcrn(band, ncla, xke=32):
    inputs = Input(shape=(5,5,band))
    x1 = Conv2D(xke*2,kernel_size=(3,3),padding='valid',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x2 = Conv2D(xke*2,kernel_size=(1,1),padding='valid',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x1 = MaxPooling2D(3,3)(x1)
    x2 = MaxPooling2D(5,5)(x2)
    x = keras.layers.concatenate([x1,x2],axis=-1)

    # residual 1
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x = Add()([x1,x])

    pre = Conv2D(ncla,kernel_size=(1,1),activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    pre = Flatten()(pre)
    model = Model(inputs=inputs, outputs=pre)
    return model

def resnet_2spa(band, ncla, xke=16):
    inputs = Input(shape=(10,10,band))
    x1 = Conv2D(xke*2,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x2 = Conv2D(xke,kernel_size=(5,5),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x3 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)

    x = keras.layers.concatenate([x1,x2,x3],axis=3)

    # residual 1
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x = Add()([x1,x])
    
    x = MaxPooling2D(2,2)(x)

    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x = Add()([x1,x])
    
    x1 = MaxPooling2D(5,5)(x)
    
    inputs2 = Input(shape=(5,5,band))
    y2 = Conv2D(xke,kernel_size=(5,5),padding='valid',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs2)
#    y2 = MaxPooling2D(3,3)(y2)

    # residual 1
    y21 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(y2)
    y21 = Activation('relu')(y21)
    y21 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(y21)
    y21 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(y21)
    y21 = Activation('relu')(y21)
    y21 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(y21)
    y2 = Add()([y21,y2])
    
    x = keras.layers.concatenate([x1,y2],axis=-1)
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*8,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*8,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    
    pre = Conv2D(ncla,kernel_size=(1,1),activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    pre = Flatten()(pre)
    model = Model(inputs=[inputs,inputs2], outputs=pre)
    return model
    

def resnet(band, ncla, xke=16):
    inputs = Input(shape=(10,10,band))
    x1 = Conv2D(xke*2,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x2 = Conv2D(xke,kernel_size=(5,5),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x3 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)

    x = keras.layers.concatenate([x1,x2,x3],axis=3)

    # residual 1
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x = Add()([x1,x])
    
    x = AveragePooling2D(2,2)(x)

    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x = Add()([x1,x])
    
    x = AveragePooling2D(5,5)(x)
    pre = Conv2D(ncla,kernel_size=(1,1),activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    pre = Flatten()(pre)
    model = Model(inputs=inputs, outputs=pre)
    return model

def resnet_adpt(band, ncla, xke=16, res=2, inx=32):
    inputs = Input(shape=(inx,inx,band))
    x1 = Conv2D(xke*2,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x2 = Conv2D(xke,kernel_size=(5,5),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x3 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)

    x = keras.layers.concatenate([x1,x2,x3],axis=3)

    # residual 1
    for i in range(res):
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1,x])
        
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(xke*8,kernel_size=(1,1),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    
    for i in range(res):
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*8,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*8,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1,x])
    
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(xke*16,kernel_size=(1,1),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    
    for i in range(res):
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*16,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*16,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1,x])
    
    x = AveragePooling2D(8,8)(x)

    pre = Conv2D(ncla,kernel_size=(1,1),activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    pre = Flatten()(pre)
    model = Model(inputs=inputs, outputs=pre)
    return model

def resnet_adpt_large(band, ncla, xke=16, res=2, inx=32):
    inputs = Input(shape=(inx,inx,band))
    x1 = Conv2D(xke*2,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x2 = Conv2D(xke,kernel_size=(5,5),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x3 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)

    x = keras.layers.concatenate([x1,x2,x3],axis=3)

    # residual 1
    for i in range(res):
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1,x])
        
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(xke*8,kernel_size=(1,1),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    
    for i in range(res):
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*8,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*8,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1,x])
    
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(xke*16,kernel_size=(1,1),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    
    for i in range(res):
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*16,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*16,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1,x])
        
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(xke*32,kernel_size=(1,1),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    
    for i in range(res):
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*32,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*32,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1,x])
    
    x = AveragePooling2D(inx//8,inx//8)(x)

    pre = Conv2D(ncla,kernel_size=(1,1),activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    pre = Flatten()(pre)
    model = Model(inputs=inputs, outputs=pre)
    return model

def resnet_num(band, ncla, xke=16, res=2):
    inputs = Input(shape=(10,10,band))
    x1 = Conv2D(xke*2,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x2 = Conv2D(xke,kernel_size=(5,5),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x3 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)

    x = keras.layers.concatenate([x1,x2,x3],axis=3)

    # residual 1
    for i in range(res):
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1, 
                                             momentum=0.9, 
                                             epsilon=0.001, 
                                             center=True, 
                                             scale=True, 
                                             beta_initializer='zeros', 
                                             gamma_initializer='ones', 
                                             moving_mean_initializer='zeros', 
                                             moving_variance_initializer='ones')(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1,x])
    
    x = AveragePooling2D(10,10)(x)
    pre = Conv2D(ncla,kernel_size=(1,1),activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    pre = Flatten()(pre)
    model = Model(inputs=inputs, outputs=pre)
    return model

def resnet_spec(band, band2, ncla, xke=16):
    inputs = Input(shape=(10,10,band))
    x1 = Conv2D(xke*2,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x2 = Conv2D(xke,kernel_size=(5,5),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)
    x3 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs)

    x = keras.layers.concatenate([x1,x2,x3],axis=3)
    
    inputs2 = Input(shape=(5,5,band2))
    x21 = Conv2D(xke*2,kernel_size=(3,3),padding='valid',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs2)
    x22 = Conv2D(xke,kernel_size=(5,5),padding='valid',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs2)
    x23 = Conv2D(xke,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inputs2)
    x21 = MaxPooling2D(3,3)(x21)
    x23 = MaxPooling2D(5,5)(x23)
    x2 = keras.layers.concatenate([x21,x22,x23],axis=3)
    
    x22 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x2)
    x22 = Activation('relu')(x22)
    x22 = Conv2D(xke*4,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x22)
    x22 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x22)
    x22 = Activation('relu')(x22)
    x22 = Conv2D(xke*4,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x22)
    x2 = Add()([x22,x2])

    # residual 1
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x = Add()([x1,x])
    
    x = AveragePooling2D(2,2)(x)

    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(3,3),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x = Add()([x1,x])
    
    x = AveragePooling2D(5,5)(x)
    x = Add()([x,x2])
    
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x1 = keras.layers.BatchNormalization(axis=-1, 
                                         momentum=0.9, 
                                         epsilon=0.001, 
                                         center=True, 
                                         scale=True, 
                                         beta_initializer='zeros', 
                                         gamma_initializer='ones', 
                                         moving_mean_initializer='zeros', 
                                         moving_variance_initializer='ones')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(xke*4,kernel_size=(1,1),padding='same',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
    x = Add()([x1,x])
    
    pre = Conv2D(ncla,kernel_size=(1,1),activation='softmax',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    pre = Flatten()(pre)
    model = Model(inputs=[inputs,inputs2], outputs=pre)
    return model