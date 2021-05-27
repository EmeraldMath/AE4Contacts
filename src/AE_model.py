
# coding: utf-8

# In[2]:


import tensorflow as tf

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL
import config as conf
import imageio
import math


# In[30]:

def encoder(latent_dim,img_sz, units_list = [1024]*5, pad_list = ['same', 'same', 'same', 'same', 'same']):
    inference_net = tf.keras.Sequential(name="encoder")
    for i, n_filter in enumerate(units_list):
        inference_net.add(tf.keras.layers.Conv2D(filters=n_filter, kernel_size=3, strides=(1, 1), padding=pad_list[i], name = "cnn_%i" % i))
        inference_net.add(tf.keras.layers.BatchNormalization(name = "batchnorm_{}".format(i)))
        inference_net.add(tf.keras.layers.Activation('selu', name = "selu_{}".format(i)))
        inference_net.add(tf.keras.layers.Dropout(0.1, name="dropout_{}".format(i)))
        inference_net.add(tf.keras.layers.MaxPool2D((2,2),(2,2), padding='same', name="maxpool_{}".format(i)))

    return inference_net
                           
                           
def decoder(latent_dim, units_list = [1024]*5, pad_list = ['same', 'same', 'same', 'same', 'same']):
    generative_net = tf.keras.Sequential(name="decoder")
    for i, n_filter in enumerate(units_list[::-1]):
        generative_net.add(tf.keras.layers.UpSampling2D((2,2), name="upsample_{}".format(i)))
        generative_net.add(tf.keras.layers.Conv2DTranspose(filters=n_filter, kernel_size=3, strides=(1, 1), padding=pad_list[len(units_list)-1-i], name = "cnnTran_{}".format(i)))
        generative_net.add(tf.keras.layers.BatchNormalization(name = "batchnorm_{}".format(i)))
        generative_net.add(tf.keras.layers.Activation('selu', name = "selu_{}".format(i)))
        generative_net.add(tf.keras.layers.Dropout(0.1, name="dropout_{}".format(i)))        
    
    generative_net.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding='same', name = "cnnTran_last"))

    return generative_net



class CVAE(tf.keras.Model):
    def __init__(self, latent_dim = 50, img_sz = conf.IMAGE_SIZE, mnist = False, units_list=[1024]*5, pad_list=['same', 'same', 'same', 'same', 'same']):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        
        if mnist == True:
            tf.print("MNIST_log_norm")
            img_sz = 28
        else:
            tf.print("CONTACT_MAP_log_norm")
                       

        self.encoder = encoder(latent_dim=latent_dim, img_sz=img_sz, units_list=units_list, pad_list=pad_list)
        self.decoder = decoder(latent_dim=latent_dim, units_list=units_list, pad_list=pad_list)                 
        self.build((None, None, None, 1))
        self.optimizer = tf.keras.optimizers.Adam(1e-4)


    def encode(self, x, training):
        z = self.encoder(x, training=training)
        return z


    def decode(self, z, training):
        l = self.decoder(z)
        l_t = tf.transpose(l, perm=[0, 2, 1, 3])
        logits = (l + l_t)/2.0
        return logits

  
    def call(self, inputs, training):
        z = self.encode(inputs, training)
        x_logit = self.decode(z, training)
        '''
        for layer in self.encoder.layers:
            print(layer.output)
        for layer in self.decoder.layers:
            print(layer.output)
        '''
        return x_logit
    

    
    #@tf.function    
    def compute_loss(self, x, training, binary):
        """ELBO assuming entries of x are binary variables, with closed form KLD."""
        z = self.encode(x, training)
        x_logit = self.decode(z, training)
        
        if binary:
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
            loss = tf.reduce_mean(cross_ent, axis=[1, 2, 3])
        else:
            batch_reconstruction = tf.reduce_sum(tf.math.square(x_logit - x), axis=[1,2])
            loss = tf.reduce_mean(batch_reconstruction, axis = 1)
            
        return loss


                       
    #@tf.function
    def compute_apply_gradients(self, x, training, binary):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(self.compute_loss(x, training, binary))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
