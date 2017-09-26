import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt 
import time
import os.path
from six.moves import xrange
from glob import glob
from operations import *


class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
        batch_size=64, sample_num=64, output_height=64, output_width=64,
        z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, channel=3, 
        dataset_name='default', input_fname_pattern='*.jpg', 
        checkpoint_dir=None, sample_dir=None):

        """
        Args:
            z_dim: dimension of the dim for z
            gf_dim: dimension of generator filters in the first deconv layer
            df_dim: dimension of discriminator filters in the first conv layer
            gfc_dim: dimension of generator units for fully connected layer
            dfc_dim: dimension of discriminator units for fully connected layer
            channel: the channel of the image

        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = os.path.join(sample_dir, self.dataset_name) 
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        # all the data path
        if self.dataset_name == 'webface':
            self.data = glob(os.path.join('./data', self.dataset_name, '*', self.input_fname_pattern))
        else:
            self.data = glob(os.path.join('./data', self.dataset_name, self.input_fname_pattern))
        self.channel = imread(self.data[0]).shape[-1]
        # if chaneel==1, it's a gray image
        self.grayscale = (self.channel == 1)

        #build the model
        self.build_model()

    def build_model(self):
        if self.crop:
            image_dims = [self.output_height, self.output_width, self.channel]
        else:
            image_dims = [self.input_height, self.input_width, self.channel]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        #inputs: [batch_size, image_h, image_w, channel]
        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.d_real, self.d_logits_real = self.discriminator(inputs, reuse=False)
        self.g = self.generator(self.z, reuse = False)
        self.sampler = self.generator(self.z, reuse = True)
        self.d_fake, self.d_logits_fake = self.discriminator(self.g, reuse=True)

        self.d_real_sum = tf.summary.histogram("d_logits_real", self.d_real)
        self.d_fake_sum = tf.summary.histogram("d_logits_fake", self.d_fake)
        self.g_result_image = tf.summary.image("g_result", self.g)



        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logits_real, labels=tf.ones_like(self.d_real)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logits_fake, labels=tf.zeros_like(self.d_fake)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logits_fake, labels=tf.ones_like(self.d_fake)))
        self.d_loss = self.d_loss_fake + self.d_loss_real

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)



        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.g_sum = tf.summary.merge([self.z_sum, self.d_fake_sum,
            self.g_result_image, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_real_sum, 
            self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

        sample_files = self.data[0:self.sample_num]
        sample = [get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
        if (self.grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            if config.dataset == 'webface': 
                self.data = glob(os.path.join("./data", config.dataset, '*', self.input_fname_pattern))
            else:
                self.data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size
            # batch_idxs = 20
            for idx in xrange(0, batch_idxs):
                batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file,
                            input_height=self.input_height,
                            input_width=self.input_width,
                            resize_height=self.output_height,
                            resize_width=self.output_width,
                            crop=self.crop,
                            grayscale=self.grayscale) for batch_file in batch_files]
                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={ self.inputs: batch_images, self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)
          
                errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
                errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))                
            
                if np.mod(counter, 100) == 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
                            feed_dict={ self.z: sample_z, self.inputs: sample_inputs})
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                    except:
                        print("one pic error!...")

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, x_input, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if(reuse):
                tf.get_variable_scope().reuse_variables()

            h0 = lrelu(conv2d(x_input, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, scope='d_h4_lin')
        return tf.nn.sigmoid(h4), h4

    def generator(self, z, reuse=False):
        with tf.variable_scope('generator') as scope:
            if(reuse):
                tf.get_variable_scope().reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            h0 = linear(z, self.gf_dim*8*s_h16*s_w16, scope='g_h0_lin')
            h0 = tf.reshape(h0, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0))

            output1_shape = [self.batch_size, s_h8, s_w8, self.gf_dim*4]
            h1, self.h1_w, self.h1_b = deconv2d(h0, output1_shape, name="g_h1", with_w=True)
            h1 = tf.nn.relu(self.g_bn1(h1))

            output2_shape = [self.batch_size, s_h4, s_w4, self.gf_dim*2]
            h2, self.h2_w, self.h2_b = deconv2d(h1, output2_shape, name="g_h2", with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            output3_shape = [self.batch_size, s_h2, s_w2, self.gf_dim]
            h3, self.h3_w, self.h3_b = deconv2d(h2, output3_shape, name="g_h3", with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            output4_shape = [self.batch_size, s_h, s_w, self.channel]    
            h4, self.h4_w, self.h4_b = deconv2d(h3, output4_shape, name="g_h4", with_w=True)
            h4 = tf.nn.tanh(h4)

        return h4

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)
      
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
