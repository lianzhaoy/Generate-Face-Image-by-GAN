import math
import numpy as np 
import tensorflow as tf
import scipy.misc
import pprint

from tensorflow.python.framework import ops

pp = pprint.PrettyPrinter()

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def conv2d(input_, output_dim,
                    k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02,
                    name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                    initializer=tf.truncated_normal_initializer(stddev=stddev))
        
        conv = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding="SAME")

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0))

        result = conv + biases
        return result

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                    initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])


        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
        return deconv, w, biases
    else:
        return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                     tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
          initializer=tf.constant_initializer(bias_start))
    if with_w:
        return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
        return tf.matmul(input_, matrix) + bias


# some functions dealing with images

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(
        x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
        image, input_height, input_width, 
        resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                        decay=self.momentum, 
                        updates_collections=None,
                        epsilon=self.epsilon,
                        scale=True,
                        is_training=train,
                        scope=self.name)