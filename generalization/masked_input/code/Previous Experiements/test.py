import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import os
import numpy as np
import random
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pylab as plt

def process_image_xy(file_path,image_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.resize(img, [image_size, image_size])/255
    
    parts = tf.strings.split(file_path, os.path.sep)
    label=tf.strings.to_number(parts[-2])
    return img,label
def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)
def apply_blur(img):
    blur = _gaussian_kernel(50, 20, 3, img.dtype)
    img = tf.nn.depthwise_conv2d(img[None], blur, [1,1,1,1], 'SAME')
    return img

image_size=224

batch_size=1

cifar100_test=tf.data.Dataset.list_files(r'..\..\pre-processing\cifar 100\test\*\*').shuffle(800).map(lambda y: process_image_xy(y,image_size))

batch_cifar100_test=cifar100_test.batch(batch_size, drop_remainder=True)

s_data,s_labels=next(iter(batch_cifar100_test))

blurred=apply_blur(s_data[0])

fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8))

axs[0,0].set_title('Baseline Image')
axs[0,0].imshow(blurred[0])
axs[0,0].axis('off')

axs[0,1].set_title('Original Image')
axs[0,1].imshow(s_data[0])
axs[0,1].axis('off') 
plt.show()
