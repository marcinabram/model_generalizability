#Implenting a method of saliency maps from https://arxiv.org/abs/1706.03825

import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import tensorflow_addons as tfa

def main():
    Model=tf.keras.models.load_model("Models/Model Base Model Rv2,50 cifar100")
    
    image_size=224
    
    batch_size=1
    
    steps=2
    
    cifar100_test=tf.data.Dataset.list_files(r'..\..\pre-processing\cifar 100\test\*\*').shuffle(800).map(lambda y: process_image_xy(y,image_size))
    
    batch_cifar100_test=cifar100_test.batch(batch_size, drop_remainder=True)
    
    s_data,blurred_image,s_labels=next(iter(batch_cifar100_test))
    
    grads=intergrated_grads(s_data[0],blurred_image[0],s_labels,Model,steps)
        
    image=tf.reduce_sum(tf.math.abs(grads),axis=-1)
    
    gradient_image=image[0]
    cmap=plt.cm.inferno
    
    fig, axs = plt.subplots(nrows=2, ncols=4, squeeze=False, figsize=(8, 8))
    
  
    axs[0,0].set_title('Original Image')
    axs[0,0].imshow(s_data[0])
    axs[0,0].axis('off') 
  
    axs[0,1].set_title('IG Attribution Mask')
    axs[0,1].imshow(gradient_image, cmap=cmap)
    axs[0,1].axis('off')  
  
    axs[0,2].set_title('Original + IG Attribution Mask Overlay')
    axs[0,2].imshow(gradient_image, cmap=cmap)
    axs[0,2].imshow(s_data[0], alpha=.4)
    axs[0,2].axis('off')

    s_map=saliency_map(s_data[0],blurred_image[0],s_labels[0],steps,Model)
    masked_image=Mask_Data(s_data,s_map,[.2],image_size)
    axs[0,3].set_title('Original + IG Attribution Mask Overlay')
    axs[0,3].imshow(masked_image)
    axs[0,3].axis('off')

    
    
    steps=50
    print("Starting Next Run\n\n")
    grads=intergrated_grads(s_data[0],blurred_image[0],s_labels,Model,steps)
        
    image=tf.reduce_sum(tf.math.abs(grads),axis=-1)
    print(image.shape)
    gradient_image=image[0]
    
  
    axs[1,0].set_title('Original Image')
    axs[1,0].imshow(s_data[0])
    axs[1,0].axis('off') 
  
    axs[1,1].set_title('IG Attribution Mask')
    axs[1,1].imshow(gradient_image, cmap=cmap)
    axs[1,1].axis('off')  
  
    axs[1,2].set_title('Original + IG Attribution Mask Overlay')
    axs[1,2].imshow(gradient_image, cmap=cmap)
    axs[1,2].imshow(s_data[0], alpha=.4)
    axs[1,2].axis('off')
    print("Starting 2\n\n\n")
    s_map=saliency_map(s_data[0],blurred_image[0],s_labels[0],steps,Model)
    print(s_map)
    masked_image=Mask_Data(s_data,s_map,[.2],image_size)
    
    axs[1,3].set_title('Original + IG Attribution Mask Overlay')
    axs[1,3].imshow(masked_image)
    axs[1,3].axis('off')

    plt.tight_layout()
  
    plt.show()
    
def intergrated_grads(Data,Baseline,Labels,Model,Steps):
    Data=tf.expand_dims(Data,axis=0)
    Baseline=tf.expand_dims(Baseline,axis=0)
    
    difference_step=(Data-Baseline)
    
    alphas = tf.linspace(start=0.0, stop=1.0, num=Steps+1)
    
    alphas = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    
    path_inputs = Baseline +  alphas * difference_step
    
    
        
    
    with tf.GradientTape() as tape_image:
        tape_image.watch(path_inputs)
        classification_vector=Model(path_inputs,training=False)
        classification_vector=tf.slice(classification_vector,[0, tf.cast(Labels[0],tf.int32)],[Steps+1,1])
        
        
    #Actually getting the gradient
    
    grads=tape_image.gradient(classification_vector, path_inputs)
    grads=(grads[:-1] + grads[1:]) / tf.constant(2.0)
    grads= tf.math.reduce_mean(grads, axis=0)
    saliency_map=difference_step*grads
    return tf.math.abs(saliency_map)

def process_image_xy(file_path,image_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img,channels=3)/255
    blurred_image=tfa.image.gaussian_filter2d(img+tf.random.normal((32,32,3),0,.01),12,16)
    
    img = tf.image.resize(img, [image_size, image_size])
    blurred_image=tf.image.resize(blurred_image, [image_size, image_size])
    
    parts = tf.strings.split(file_path, os.path.sep)
    label=tf.strings.to_number(parts[-2])
    return img,blurred_image,label

def saliency_map(Data,Baseline,Labels,Steps,Model):
    Data=tf.expand_dims(Data,axis=0)
    Baseline=tf.expand_dims(Baseline,axis=0)
    difference_step=(Data-Baseline)
    
    alphas = tf.linspace(start=0.0, stop=1.0, num=Steps+1)
    
    alphas = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    
    path_inputs = Baseline +  alphas * difference_step
    
    
        
    
    with tf.GradientTape() as tape_image:
        tape_image.watch(path_inputs)
        classification_vector=Model(path_inputs,training=False)
        classification_vector=tf.slice(classification_vector,[0, tf.cast(Labels,tf.int32)],[Steps+1,1])
        
        
    #Actually getting the gradient
    
    grads=tape_image.gradient(classification_vector, path_inputs)
    grads=(grads[:-1] + grads[1:]) / tf.constant(2.0)
    grads= tf.math.reduce_mean(grads, axis=0)
    saliency_map=difference_step*grads
    return tf.math.abs(saliency_map)

def Mask_Data(Data,saliency_map,percent,image_size):
    
    #gradients=gaussian_filter(gradients, sigma=1)
    
    #flattening the image makes it easier to find the max
    line_map = tf.keras.layers.Flatten()(saliency_map)
    #Normalizing everything
    line_map= tf.math.divide(line_map,tf.math.reduce_sum(line_map,axis=1))
    
    #Sorting the map in increasing order
    sorted_saliency_order=tf.argsort(-1*line_map)
    sorted_saliency=tf.gather(line_map,sorted_saliency_order, batch_dims=-1)
    
    #Masking the largest percent
    cum_sum=tf.math.cumsum(sorted_saliency,axis=1)
    Data=tf.keras.layers.Flatten()(Data)
    
    Data_shuffled=tf.zeros_like(Data)
    
    Data_Masked=Apply_Mask(Data[0],cum_sum[0],sorted_saliency_order[0],Data_shuffled[0],percent)
    
    Data=tf.reshape(Data_Masked,[image_size,image_size,3])

    return Data
def Apply_Mask(Data,cum_sum,sorted_saliency_order,Data_shuffled,percent):
    amount_to_mask=tf.searchsorted(cum_sum,percent)
    indices=tf.slice(sorted_saliency_order,[0],amount_to_mask)
    indices=tf.reshape(indices,[indices.shape[0]])
    updates=tf.gather(Data_shuffled,indices)
    updates=tf.cast(indices,dtype=tf.float32)
    indices=tf.expand_dims(indices,axis=1)
    Data_Masked=tf.tensor_scatter_nd_update(Data, indices, updates)
    return Data_Masked
    
if __name__=='__main__':
    main()