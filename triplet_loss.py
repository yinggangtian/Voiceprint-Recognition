import logging
import keras
import keras.backend as K
import constants as c
import keras.layers
import numpy as np
import math
from tensorflow import keras
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
import random
'''
https://arxiv.org/pdf/1801.07698.pdf
https://github.com/deepinsight/insightface
https://github.com/auroua/InsightFace_TF/blob/master/losses/face_losses.py

'''

alpha = c.ALPHA  # used in FaceNet https://arxiv.org/pdf/1503.03832.pdf


def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    dot = K.squeeze(K.batch_dot(x1, x2, axes=1), axis=1)
    return dot

def center_loss(num_classes:int):

    def center_loss_(labels, features):
        """
        获取center loss及更新样本的center
        :param labels: Tensor,表征样本label,非one-hot编码,shape应为(batch_size,).
        :param features: Tensor,表征样本特征,最后一个fc层的输出,shape应该为(batch_size, num_classes).
        :param alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        :param num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
        :return: Tensor, center-loss， shape因为(batch_size,)
        """
        #根据网络的输出神经元数量
        # 更新中心的学习率
        alpha = 0.6
        
        # 获取特征的维数，例如256维
        len_features = features.get_shape()[1]
        # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
        # 设置trainable=False是因为样本中心不是由梯度进行更新的
        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                                initializer=tf.constant_initializer(0), trainable=False)
        # 将label展开为一维的，如果labels已经是一维的，则该动作其实无必要
        labels = tf.reshape(labels, [-1])

        # 根据样本label,获取mini-batch中每一个样本对应的中心值
        centers_batch = tf.gather(centers, tf.cast(labels, tf.int32))

        # 当前mini-batch的特征值与它们对应的中心值之间的差
        diff = centers_batch - features

        # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff

        # 更新centers
        centers_update_op = tf.scatter_sub(centers, tf.cast(labels, tf.int32), diff)

        # 这里使用tf.control_dependencies更新centers
        with tf.control_dependencies([centers_update_op]):
            # 计算center-loss
            c_loss = tf.nn.l2_loss(features - centers_batch)

        return c_loss
    return center_loss_



def coco_loss(out_num:int):
    w_init = tf.contrib.layers.xavier_initializer(uniform=False)
    s = 30
    m = 0.4
    def cosineface_losses(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        with tf.variable_scope('coco_losss'):
            y_pred_norm = tf.norm(y_pred, axis=1 ,keep_dims=True)
            y_pred = tf.div(y_pred, y_pred_norm, name='norm_ypred')

            weights = tf.get_variable(name='embedding_weights', shape=(y_pred.shape[-1], out_num),
                                    initializer=w_init, dtype=tf.float32)
            weights_norm = tf.norm(weights, axis=0, keep_dims=True)
            weights = tf.div(weights, weights_norm, name='norm_weights')
            
            # cos_theta - m
            cos_t = tf.matmul(y_pred, weights, name='cos_t')
            cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')
            
            mask = tf.one_hot(y_true, depth=out_num, name='one_hot_mask')
            inv_mask = tf.subtract(1., mask, name='inverse_mask')
            
            output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='coco_loss_output')
        
        return output

    return cosineface_losses


def softmax_loss(out_num:int):

    def softmax_loss_(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.one_hot(y_true, depth=out_num, name='one_hot_mask')

        y = keras.activations.softmax(y_pred, axis=-1)
        loss = keras.losses.categorical_crossentropy(one_hot, y)
        return loss

    return softmax_loss_


def AAM_loss(out_num: int):
    w_init = tf.contrib.layers.xavier_initializer(uniform=False)
    s, m = 64, 0.5

    def additive_angular_margin_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = sin_m * m  # issue 1
        threshold = math.cos(math.pi - m)

        with tf.variable_scope('aam_loss'):
            # inputs and weights norm
            y_pred_norm = tf.norm(y_pred, axis=1, keep_dims=True)
            y_pred = tf.div(y_pred, y_pred_norm, name='norm_y_pred')
            weights = tf.get_variable(name='embedding_weights', shape=(y_pred.shape[-1], out_num),
                                    initializer=w_init, dtype=tf.float32)
            weights_norm = tf.norm(weights, axis=0, keep_dims=True)
            weights = tf.div(weights, weights_norm, name='norm_weights')
            # cos(theta+m)
            cos_t = tf.matmul(y_pred, weights, name='cos_t')
            cos_t2 = tf.square(cos_t, name='cos_2')
            sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
            sin_t = tf.sqrt(sin_t2, name='sin_t')
            cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

            # this condition controls the theta+m should in range [0, pi]
            #      0<=theta+m<=pi
            #     -m<=theta<=pi-m
            cond_v = cos_t - threshold
            cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

            keep_val = s*(cos_t - mm)
            cos_mt_temp = tf.where(cond, cos_mt, keep_val)

            mask = tf.one_hot(y_true, depth=out_num, name='one_hot_mask')
            # mask = tf.squeeze(mask, 1)
            inv_mask = tf.subtract(1., mask, name='inverse_mask')

            s_cos_t = s * cos_t

            output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='aam_loss_output')
        return output
    return additive_angular_margin_loss
        

def sigmoid_cross_entropy_loss(out_num:int):

    def CE_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.one_hot(y_true, depth=out_num, name='one_hot_mask')
        loss = keras.losses.categorical_crossentropy(one_hot, y_pred)
        return loss

    return CE_loss



def cross_entropy_loss(out_num:int):

    def CE_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.one_hot(y_true, depth=out_num, name='one_hot_mask')
        loss = keras.losses.categorical_crossentropy(one_hot, y_pred)
        return loss

    return CE_loss



def deep_speaker_loss(y_true, y_pred):
    # y_true.shape = (batch_size, embedding_size)
    # y_pred.shape = (batch_size, embedding_size)
    # CONVENTION: Input is:
    # concat(BATCH_SIZE * [ANCHOR, POSITIVE_EX, NEGATIVE_EX] * NUM_FRAMES)
    # EXAMPLE:
    # BATCH_NUM_TRIPLETS = 3, NUM_FRAMES = 2
    # _____________________________________________________
    # ANCHOR 1 (512,)
    # ANCHOR 2 (512,)
    # ANCHOR 3 (512,)
    # POS EX 1 (512,)
    # POS EX 2 (512,)
    # POS EX 3 (512,)
    # NEG EX 1 (512,)
    # NEG EX 2 (512,)
    # NEG EX 3 (512,)
    # _____________________________________________________

    #elements = int(y_pred.shape.as_list()[0] / 3)
    elements = c.BATCH_SIZE

    anchor = y_pred[0:elements]
    positive_ex = y_pred[elements:2 * elements]
    negative_ex = y_pred[2 * elements:]

    sap = batch_cosine_similarity(anchor, positive_ex)
    san = batch_cosine_similarity(anchor, negative_ex)
    loss = K.maximum(san - sap + alpha, 0.0)
    total_loss = K.sum(loss)
    return total_loss



if __name__ == "__main__":
    a = np.array([1,3,1,2])
    b = np.array([0.3,0.2,0.1,0.5])
    
    # loss = cross_entropy_loss(a, b)
    # print(loss)
