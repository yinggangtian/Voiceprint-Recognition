import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from switchable_norm import SwitchNormalization
import tensorflow as tf
from tensorflow.keras import backend as K, layers, regularizers, Model
from tensorflow.keras.layers import (
    Input, GRU, Conv2D, Lambda, Dense, RepeatVector, Permute, Reshape, BatchNormalization, Dropout, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
import constants as c

def identity_block2(input_tensor, kernel_size, filters, stage, block):  # next step try full-pre activation
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = Conv2D(filters,
               kernel_size=1,
               strides=1,
               activation=None,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(0.00001),
               name=conv_name_base + '_conv1_1')(input_tensor)
    # x = BatchNormalization(name=conv_name_base + '_conv1.1_bn')(x)
    x = SwitchNormalization(axis=-1)(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
               kernel_size=kernel_size,
               strides=1,
               activation=None,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(0.00001),
               name=conv_name_base + '_conv3')(x)
    # x = BatchNormalization(name=conv_name_base + '_conv3_bn')(x)
    x = SwitchNormalization(axis=-1)(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
               kernel_size=1,
               strides=1,
               activation=None,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(0.00001),
               name=conv_name_base + '_conv1_2')(x)
    # x = BatchNormalization(name=conv_name_base + '_conv1.2_bn')(x)
    x = SwitchNormalization(axis=-1)(x)

    x = layers.add([x, input_tensor])
    x = clipped_relu(x)
    return x


def clipped_relu(x):
    return tf.keras.layers.Lambda(
        lambda y: tf.minimum(tf.maximum(y, 0.), 20.)
    )(x)


def identity_block(x, kernel_size, filters, stage, block):
    name = f"res{stage}_{block}"
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(1e-5),
               name=name+"_2a")(x)
    x = SwitchNormalization(axis=-1)(x)
    x = clipped_relu(x)

    x = Conv2D(filters, kernel_size, padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(1e-5),
               name=name+"_2b")(x)
    x = SwitchNormalization(axis=-1)(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = clipped_relu(x)
    return x

def convolutional_model(input_shape=(c.NUM_MELS, c.NUM_FRAMES, c.CHANNELS)):
    inp = Input(shape=input_shape, name="mel_input")  # (64, None, 1)

    # 下采样 4 次，每次 mel 维和 time 维都 /2
    x = inp
    for i, f in enumerate([64, 128, 256, 512], start=1):
        # stride=2 的卷积
        x = Conv2D(f, kernel_size=5, strides=2, padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(1e-5),
                   name=f"conv{i}_{f}s")(x)
        x = SwitchNormalization(axis=-1)(x)
        x = clipped_relu(x)
        # 三个残差块
        for j in range(3):
            x = identity_block(x, 3, f, stage=i, block=j)

    # 此时 x.shape = (batch, mel_blocks=4, time_steps, 512)

    # 1) 把时间轴移到第二维
    x = Permute((2, 1, 3), name="permute_time_mel_channel")(x)
    # 2) 合并 mel_blocks × channels → 2048
    FEATURE_DIM = (c.NUM_MELS // 16) * 512  # = 4 * 512
    x = Reshape(target_shape=(None, FEATURE_DIM), name="reshape_to_2048")(x)
    # 3) 对 time_steps 进行全局平均
    x = GlobalAveragePooling1D(name="global_avg_time")(x)
    # 4) 全连接 + L2 归一化
    x = Dense(512, name="affine")(x)
    x = Lambda(lambda y: tf.math.l2_normalize(y, axis=1), name="ln")(x)

    model = Model(inputs=inp, outputs=x, name="convolutional")
    model.summary()
    return model

def convolutional_model_simple(input_shape=(c.NUM_FRAMES,64, 1),    #input_shape(32,32,3)
                             batch_size=c.BATCH_SIZE * c.TRIPLET_PER_BATCH , num_frames=c.NUM_FRAMES):
    # http://cs231n.github.io/convolutional-networks/
    # conv weights
    # #params = ks * ks * nb_filters * num_channels_input

    # Conv128-s
    # 5*5*128*128/2+128
    # ks*ks*nb_filters*channels/strides+bias(=nb_filters)

    # take 100 ms -> 4 frames.
    # if signal is 3 seconds, then take 100ms per 100ms and average out this network.
    # 8*8 = 64 features.

    # used to share all the layers across the inputs

    # num_frames = K.shape() - do it dynamically after.

    def conv_and_res_block(inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = Conv2D(filters,
                   kernel_size=5,
                   strides=2,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(0.00001), name=conv_name)(inp)
        # o = BatchNormalization(name=conv_name + '_bn')(o)
        o = SwitchNormalization(axis=-1)(o)
        o = clipped_relu(o)
        for i in range(3):
            o = identity_block2(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(inp):
        x_ = conv_and_res_block(inp, 64, stage=1)
        x_ = conv_and_res_block(x_, 128, stage=2)
        x_ = conv_and_res_block(x_, 256, stage=3)
        #x_ = conv_and_res_block(x_, 512, stage=4)
        return x_

    inputs = tf.keras.Input(shape=input_shape)  # TODO the network should be definable without explicit batch shape
    x = cnn_component(inputs)  # .shape = (BATCH_SIZE , num_frames/8, 64/8, 512)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames / 8), 2048)), name='reshape')(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)  #shape = (BATCH_SIZE, 512)
    x = Dense(512, name='affine')(x)  # .shape = (BATCH_SIZE , 512)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)

    model = Model(inputs, x, name='convolutional')
    print(model.summary())
    return model



def recurrent_model(input_shape=(c.NUM_MELS, c.NUM_FRAMES, c.CHANNELS)):
    inp = Input(shape=input_shape, name="mel_input")

    # 一个下采样块示例：64→32
    x = Conv2D(64, kernel_size=5, strides=2, padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
               name="conv_downsample")(inp)
    x = SwitchNormalization(axis=-1)(x)
    x = tf.keras.layers.Lambda(lambda y: tf.minimum(tf.maximum(y, 0.), 20.))(x)

    # 此时 x.shape = (batch, mel_blocks=32, time_frames/2, 64)
    # 如果要对 mel_blocks 和 channel 合并，同样使用 Permute+Reshape：
    # 先 Permute：(batch, time, mel, chan)
    x = Permute((3, 2, 1), name="permute_time_mel_channel")(x)
    # 再 Reshape：(batch, time, mel*chan)
    MEL_BLOCKS = c.NUM_MELS // 2  # 64/2=32
    FEATURE_DIM = MEL_BLOCKS * 64
    x = Reshape(target_shape=(None, FEATURE_DIM), name="reshape_to_feat")(x)

    # 接三层 GRU
    x = GRU(1024, return_sequences=True, name="gru1")(x)
    x = GRU(1024, return_sequences=True, name="gru2")(x)
    x = GRU(1024, return_sequences=True, name="gru3")(x)

    # 再全局平均
    x = GlobalAveragePooling1D(name="global_avg_time")(x)
    x = Dropout(0.5, name="dropout")(x)
    x = Dense(512, name="affine")(x)

    model = Model(inputs=inp, outputs=x, name="recurrent")
    model.summary()
    return model

#=======================================================================================
'''
下面是新加的loss，所对应的模型
'''
#=======================================================================================
def recurrent_model_softmax(input_shape=(c.NUM_FRAMES, 64, 1),
                            batch_size=c.BATCH_SIZE * c.TRIPLET_PER_BATCH ,num_frames=c.NUM_FRAMES, num_spks=c.NUM_SPEAKERS):
    inputs = Input(shape=input_shape)
    #x = Permute((2,1))(inputs)
    x = Conv2D(64,kernel_size=5,strides=2,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
    # x = BatchNormalization()(x)  #shape = (BATCH_SIZE , num_frames/2, 64/2, 64)

    x = SwitchNormalization(axis=-1)(x)
    x = clipped_relu(x)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames / 2), 2048)), name='reshape')(x) #shape = (BATCH_SIZE , num_frames/2, 2048)
    x = GRU(1024,return_sequences=True,dropout=0.1,recurrent_dropout=0.5)(x)  #shape = (BATCH_SIZE , num_frames/2, 1024)
    x = GRU(1024,return_sequences=True,dropout=0.1,recurrent_dropout=0.5)(x)
    x = GRU(1024,return_sequences=True,dropout=0.1,recurrent_dropout=0.5)(x)  #shape = (BATCH_SIZE , num_frames/2, 1024)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x) #shape = (BATCH_SIZE, 1024)

    x = Dropout(0.5)(x)
    x = Dense(512)(x)  #shape = (BATCH_SIZE, 512)
    x = clipped_relu(x)
    x = Dropout(0.5)(x)

    x = Dense(num_spks, activation='softmax')(x)

    model = Model(inputs, x, name='recurrent_softmax')

    print(model.summary())
    return model

def recurrent_model_sigmoid_cross_entropy(input_shape=(c.NUM_FRAMES, 64, 1),
                                         batch_size=c.BATCH_SIZE * c.TRIPLET_PER_BATCH ,num_frames=c.NUM_FRAMES, num_spks=c.NUM_SPEAKERS):
    '''
    6.	Sigmoid cross entropy loss
    '''
    inputs = Input(shape=input_shape)
    #x = Permute((2,1))(inputs)
    x = Conv2D(64,kernel_size=5,strides=2,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
    # x = BatchNormalization()(x)  #shape = (BATCH_SIZE , num_frames/2, 64/2, 64)

    x = SwitchNormalization(axis=-1)(x)
    x = clipped_relu(x)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames / 2), 2048)), name='reshape')(x) #shape = (BATCH_SIZE , num_frames/2, 2048)
    x = GRU(1024,return_sequences=True,dropout=0.1,recurrent_dropout=0.5)(x)  #shape = (BATCH_SIZE , num_frames/2, 1024)
    x = GRU(1024,return_sequences=True,dropout=0.1,recurrent_dropout=0.5)(x)
    x = GRU(1024,return_sequences=True,dropout=0.1,recurrent_dropout=0.5)(x)  #shape = (BATCH_SIZE , num_frames/2, 1024)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x) #shape = (BATCH_SIZE, 1024)

    x = Dropout(0.5)(x)
    x = Dense(512, activation='sigmoid')(x)  #shape = (BATCH_SIZE, 512)
    x = Dropout(0.5)(x)

    x = Dense(num_spks)(x)
    model = Model(inputs, x, name='recurrent_Sigmoid_cross_entropy')

    print(model.summary())
    return model


def recurrent_model_cross_entropy(input_shape=(c.NUM_FRAMES, 64, 1),
                                  batch_size=c.BATCH_SIZE * c.TRIPLET_PER_BATCH ,num_frames=c.NUM_FRAMES, num_spks=c.NUM_SPEAKERS):
    '''
    6.	Sigmoid cross entropy loss（有API库）
    '''
    inputs = Input(shape=input_shape)
    #x = Permute((2,1))(inputs)
    x = Conv2D(64,kernel_size=5,strides=2,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
    # x = BatchNormalization()(x)  #shape = (BATCH_SIZE , num_frames/2, 64/2,