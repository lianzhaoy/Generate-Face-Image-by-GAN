
# coding: utf-8

# Loading MNIST Data

import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt 


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")


x_train = mnist.train.images[:55000, :]
x_train.shape


# Let's look at what a random image might look like

randomNum = random.randint(0, 55000)
image = x_train[randomNum].reshape([28,28])
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()


# define some useful functions

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# Discriminator

def discriminator(x_input, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        #First conv and pool layers
        W_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
        h_conv1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1)
        ## batch_size * (28,28) * 8 ==> batch_size * (14,14) * 8
        h_pool1 = avg_pool_2x2(h_conv1)
        
        #Second conv and pool layers
        W_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        ## batch_size * ((14,14) * 8 ==> batch_size * (7,7) * 16
        h_pool2 = avg_pool_2x2(h_conv2)
        
        #First fully connected layer
        W_fc1 = tf.get_variable('d_wfc1', [7*7*16, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
        ## batch_size * (7,7) * 16 ==> batch_size * 32
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        #Second fully connected layer and output
        W_fc2 = tf.get_variable('d_wfc2', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))
        y_output = tf.matmul(h_fc1, W_fc2) + b_fc2
        
    return y_output


# Generator

# Which takes in a noise vector and upsample it to become a 28 * 28 image

def generator(z, batch_size, z_dim, reuse=False):
    with tf.variable_scope('generator') as scope:
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        g_dim = 64 #Number of filters of first layer of generator
        c_dim = 1 #Number of chanel
        s = 28
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        ## s2, s4, s8, s16 = 14, 7, 3, 1

        h0 = tf.reshape(z, [batch_size, s16+1, s16+1, 25])
        h0 = tf.nn.relu(h0)

        #First trans conv layer
        output1_shape = [batch_size, s8, s8, g_dim*4]
        W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1]) ], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(0.1))
        H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, strides=[1, 2, 2, 1], padding='SAME')
        H_conv1 = tf.contrib.layers.batch_norm(inputs = H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
        H_conv1 = tf.nn.relu(H_conv1)
        # batch_size * (3, 3) * 256

        #Second trans conv layer
        output2_shape = [batch_size, s4-1, s4-1, g_dim*2]
        W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(0.1))
        H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1], padding='SAME')
        H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
        H_conv2 = tf.nn.relu(H_conv2)
        # batch_size * (6, 6) * 128

        #Third trans conv layer
        output3_shape = [batch_size, s2-2, s2-2, g_dim]
        W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(0.1))
        H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1], padding='SAME')
        H_conv3 = tf.contrib.layers.batch_norm(inputs = H_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
        H_conv3 = tf.nn.relu(H_conv3)
        # batch_size * (12, 12) * 64   

        #Fourth DeConv Layer
        output4_shape = [batch_size, s, s, c_dim]
        W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(0.1))
        H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1], padding='VALID')
        H_conv4 = tf.nn.tanh(H_conv4)
        # batch_size * (28, 28) * 1      
    return H_conv4


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
    trainerD = tf.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
    trainerG = tf.train.AdamOptimizer().minimize(g_loss, var_list = g_vars)


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




