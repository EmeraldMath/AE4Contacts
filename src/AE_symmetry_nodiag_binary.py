#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL
import config as conf
import imageio


# In[2]:


import AE_model as pcvae

# In[3]:


parent_dir = '/cavern/bihuayu/protein_DP/AE_symmetry/nodiag/binary/'


# In[6]:


def normalization(x):
    sz = tf.size(x['contact'])
    edge = tf.math.sqrt(tf.cast(sz, tf.float32))
    edge = tf.cast(edge, tf.int32)
    
    contact_img = x['contact']
    
    diag = tf.linalg.diag_part(x['contact'])
    x['diag'] = tf.linalg.diag(diag)
    x['contact'] = x['contact'] - x['diag']
    
    zero = tf.constant(0, dtype=tf.float32)
    one = tf.constant(1, dtype=tf.float32)
    img = tf.where(tf.not_equal(x["contact"], zero), one, zero)
    
    img_norm = tf.expand_dims(img, -1)
    
    if edge > 600:
        img_norm = tf.image.resize_with_crop_or_pad(img_norm, 600, 600)
    elif edge < 8:
        img_norm = tf.image.resize_with_crop_or_pad(img_norm, 8, 8)
    elif edge % 8 != 0:
        mod = 8 - edge % 8
        img_norm = tf.image.resize_with_crop_or_pad(img_norm, edge+mod, edge+mod)
        
    x['contact'] = img_norm
    x['diag'] = img_norm
        
    return x


# In[7]:


TRAIN_BUF = 6000
BATCH_SIZE = 1
TEST_BUF = 1000

import contact_map as cp
data = cp.contact_map()

train_norm = data.train.map(normalization)
val_varseq_norm = data.val_varseq.map(normalization)
val_decoy_norm = data.val_decoy.map(normalization)

train_set = train_norm.shuffle(TRAIN_BUF).batch(BATCH_SIZE)
val_varseq_set = val_varseq_norm.batch(1)
val_decoy_set = val_decoy_norm.shuffle(TRAIN_BUF).batch(BATCH_SIZE)


# In[8]:


for x in val_varseq_set:
    plt.imshow(x['contact'][0,:,:,0])
    break


# In[9]:


epochs = 50

model = pcvae.CVAE(latent_dim = 50, mnist = False, units_list=[64,128,128])
model.compile(optimizer = model.optimizer, loss = model.compute_loss, metrics = ["accuracy"] )
model.summary()


# In[ ]:


def generate_and_display_images(model, test_input):
    z = model.encode(test_input, False)
    recon = model.decode(z, False)
    fig = plt.figure(figsize=(4,4))
        
    for i in range(1):
        plt.subplot(2, 1, i+1)
        tf.print('orignal image')
        plt.imshow(test_input[i, :, :, 0])
        plt.axis('off')
        tf.print('reconstruction')
        plt.subplot(2, 1, i+2)
        plt.imshow(recon[i, :, :, 0])
        plt.axis('off')


    plt.savefig(parent_dir + 'image_at_epoch_{:04d}.png'.format(epoch))        
#     plt.show()
    plt.close()


checkpoint_path = parent_dir + "cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

train_loss = []
val_loss = []

for epoch in range(1, epochs + 1):
    t_loss = tf.keras.metrics.Mean()
    loss = tf.keras.metrics.Mean()
    score = []
    val_varseq_predict = []
    val_varseq_label = []

    start_time = time.time()
    for train_x in train_set:
        assert not np.any(np.isnan(train_x["contact"]))
        cost = model.compute_apply_gradients(train_x["contact"], True, True)
        t_loss(cost)
    end_time = time.time()


    print("validation")
    for test_x in val_varseq_set:
        assert not np.any(np.isnan(test_x["contact"]))
        score = model.compute_loss(test_x["contact"], False, True)
        loss(score)
        val_varseq_predict = tf.concat([val_varseq_predict, score], 0)
        val_varseq_label = tf.concat([val_varseq_label, test_x["label"]], 0)
  
    print('Epoch: {}, Test set loss: {}, '
        'time elapse for current epoch {}'.format(epoch,
                                                  loss.result(),
                                                  end_time - start_time))

    train_loss.append(t_loss.result())
    val_loss.append(loss.result())
    model.save_weights(checkpoint_path.format(epoch=epoch))
    generate_and_display_images(model, test_x["contact"])


# In[ ]:


import pickle
f = open(parent_dir + "train_loss.pkl","wb")
pickle.dump(train_loss,f)
f.close()

f = open(parent_dir + "val_loss.pkl","wb")
pickle.dump(val_loss,f)
f.close()


# In[ ]:




