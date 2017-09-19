
# coding: utf-8

# Loading MNIST Data

import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt 
from glob import glob
import os.path
from operations import *
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/")

# Let's look at what a random image might look like

randomNum = random.randint(0, 100)
image_path = glob(os.path.join("./data", "celebA", "*.jpg"))
c_dim = imread(image_path[0]).shape[-1]
grayscale = (c_dim == 1)
print(image_path[0])
image = get_image(image_path[0], input_height=108,
                    input_width=108,
                    resize_height=64,
                    resize_width=64,
                    crop=False,
                    grayscale=grayscale)
plt.imshow(image,cmap=plt.cm.jet)
plt.show()



# Discriminator
df_dim = 64
d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

def discriminator(x_input, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if(reuse):
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(x_input, df_dim, name='d_h0_conv'))
        h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
        h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [h3.get_shape()[0], -1]), 1, name='d_h4_lin')
        #First conv and pool layers
        ## batch_size * (28,28) * 8 ==> batch_size * (14,14) * 8
        
        #Second conv and pool layers
        ## batch_size * ((14,14) * 8 ==> batch_size * (7,7) * 16
        
        #First fully connected layer
        ## batch_size * (7,7) * 16 ==> batch_size * 32
        #Second fully connected layer and output
    return tf.nn.sigmoid(h4)


# Generator

# Which takes in a noise vector and upsample it to become a 28 * 28 image
gf_dim = 64
output_height=64
output_width=64
c_dim = 3 #Number of chanel
g_bn0 = batch_norm(name='g_bn1')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')

def generator(z, batch_size, z_dim, reuse=False):
    with tf.variable_scope('generator') as scope:
        if(reuse):
            tf.get_variable_scope().reuse_variables()

        s_h, s_w = output_height, output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = linear(z, gf_dim*8*s_h16*s_w16, name='g_h0_lin')
        h0 = tf.reshape(h0, [-1, s_h16, s_w16, gf_dim * 8])
        h0 = tf.nn.relu(g_bn0(h0))

        #First trans conv layer
        output1_shape = [batch_size, s_h8, s_w8, gf_dim*4]
        # batch_size * (3, 3) * 256
        h1 = deconv2d(h0, output1_shape, name="g_h1")
        h1 = tf.nn.relu(g_bn1(h1))

        #Second trans conv layer
        output2_shape = [batch_size, s_h4, s_w4, g_dim*2]
        # batch_size * (6, 6) * 128
        h2 = deconv2d(h1, output2_shape, name="g_h2")
        h2 = tf.nn.relu(g_bn2(h2))

        #Third trans conv layer
        output3_shape = [batch_size, s_h2, s_w2, g_dim]
        # batch_size * (12, 12) * 64   
        h3 = deconv2d(h2, output3_shape, name="g_h3")
        h3 = tf.nn.relu(g_bn3(h3))

        #Fourth DeConv Layer
        output4_shape = [batch_size, s_h, s_w, c_dim]
        # batch_size * (28, 28) * 1     
        h4 = deconv2d(h3, output4_shape, name="g_h4")
        h4 = tf.nn.tanh(h4)

    return h4


# Generating a sample image with untrained generator

z_dim = 100
z_test = tf.placeholder(tf.float32, [None, z_dim])
sample_image = generator(z_test, 1, z_dim)
test_z = np.random.normal(-1, 1, [1, z_dim])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    temp = sess.run(sample_image, feed_dict = {z_test : test_z})
    my_image = temp.squeeze()
    plt.imshow(my_image, cmap='gray_r')
    plt.show()


# ## Train the GAN


batch_size = 16
tf.reset_default_graph()

sess= tf.Session()
x_placeholder = tf.placeholder("float", shape=[None, 28, 28, 1])
z_placeholder = tf.placeholder(tf.float32, [None, z_dim])



Dx = discriminator(x_placeholder) # real
Gz = generator(z_placeholder, batch_size, z_dim, reuse=False) # generator
Dg = discriminator(Gz, reuse=True) #fake



g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))



d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dx)))
d_loss = d_loss_real + d_loss_fake



tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]



with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(d_loss, var_list = d_vars)
    trainerG = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(g_loss, var_list = g_vars)


# ### Training loop


sess.run(tf.global_variables_initializer())
iterations = 100
for i in range(iterations):
    z_batch = np.random.normal(-1, 1, size = [batch_size, z_dim])
    real_image_batch, image_labels = mnist.train.next_batch(batch_size)
    real_image_batch = np.reshape(real_image_batch, [batch_size, 28, 28, 1])
    _, dLoss = sess.run([trainerD, d_loss], feed_dict={z_placeholder:z_batch, x_placeholder:real_image_batch})
    _, gLoss = sess.run([trainerG, g_loss], feed_dict={z_placeholder:z_batch})



sample_image = generator(z_placeholder, 1, z_dim, reuse=True)
z_batch = np.random.normal(-1, 1, size=[1, z_dim])
temp = (sess.run(sample_image, feed_dict={z_placeholder: z_batch}))
my_i = temp.squeeze()
plt.imshow(my_i, cmap='gray_r')




