#! /usr/bin/python
# -*- coding: utf8 -*-
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from config import *



def lrelu(x):
    return tf.maximum(x*0.2,x)



def LapSRNSingleLevel(net_image, net_feature, reuse=False):
    with tf.variable_scope("Model_level", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        
        net_tmp = net_feature
        # recursive block
        for d in range(config.model.resblock_depth):
            net_tmp = PReluLayer(net_tmp, name='prelu_D%s'%(d))
            net_tmp = Conv2dLayer(net_tmp,shape=[3,3,64,64],strides=[1,1,1,1],
                        name='conv_D%s'%(d), W_init=tf.contrib.layers.xavier_initializer())
                # recursive_scope.reuse_variables()
                # for r in range(1,config.model.recursive_depth):
                #     # recursive block
                #     for d in range(config.model.resblock_depth):
                #         net_tmp = Conv2dLayer(net_tmp,shape=[3,3,64,64],strides=[1,1,1,1],
                #                             act=lrelu,name='Level%s_D%s_conv'%(level,d))
        net_feature = ElementwiseLayer(layer=[net_feature,net_tmp],combine_fn=tf.add,name='add_feature')

        net_feature = PReluLayer(net_feature, name='prelu_feature')
        net_feature = Conv2dLayer(net_feature,shape=[3,3,64,256],strides=[1,1,1,1],
                        name='upconv_feature', W_init=tf.contrib.layers.xavier_initializer())
        net_feature = SubpixelConv2d(net_feature,scale=2,n_out_channel=64,
                        name='subpixel_feature')

        # add image back
        gradient_level = Conv2dLayer(net_feature,shape=[3,3,64,3],strides=[1,1,1,1],
                        name='grad', W_init=tf.contrib.layers.xavier_initializer())
        net_image = Conv2dLayer(net_image,shape=[3,3,3,12],strides=[1,1,1,1],
                        name='upconv_image', W_init=tf.contrib.layers.xavier_initializer())
        net_image = SubpixelConv2d(net_image,scale=2,n_out_channel=3,
                        name='subpixel_image')
        net_image = ElementwiseLayer(layer=[gradient_level,net_image],combine_fn=tf.add,name='add_image')
    
    return net_image, net_feature, gradient_level



def LapSRN(inputs, is_train=False, reuse=False):
    n_level = int(np.log2(config.model.scale))
    assert n_level >= 1

    with tf.variable_scope("LapSRN", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        
        shapes = tf.shape(inputs)
        inputs_level = InputLayer(inputs, name='input_level')
            
        net_feature = Conv2dLayer(inputs_level, shape=[3,3,3,64], strides=[1,1,1,1], 
                        W_init=tf.contrib.layers.xavier_initializer(), 
                        name='init_conv')
        net_image = inputs_level

        # net_image, net_feature, net_gradient = LapSRNSingleLevel(net_image, net_feature, reuse=reuse)
        # for level in range(1,n_level):
            # net_image, net_feature, net_gradient = LapSRNSingleLevel(net_image, net_feature, reuse=True)

        net_image1, net_feature1, net_gradient1 = LapSRNSingleLevel(net_image, net_feature, reuse=reuse)
        net_image2, net_feature2, net_gradient2 = LapSRNSingleLevel(net_image1, net_feature1, reuse=True)

    return net_image2, net_gradient2, net_image1, net_gradient1