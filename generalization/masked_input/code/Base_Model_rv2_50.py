import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import os
import numpy as np
import random
from PIL import Image
from scipy.ndimage import gaussian_filter

#This file runs through one epoch for training
def main():
    
    #Model Parameters
    image_size=224
    
    batch_size=64
    
    #weight_decay_param=.0001W
    
    from_previous_model=0
    
    number_of_epochs=100
    
    learning_rate=3*(10**(-4))
    
    #Directory names
    
    model_name="Base Model Rv2,50 cifar100"
    
    data_directory="Data/"+model_name
    
    
    
    #Making directories to store stuff
    make_directory("Models")
    
    make_directory("Data")
    
    make_directory(data_directory)
        
    
    #Loading a previous model
    if(from_previous_model==1):
        
        #Loading the model
        Model=tf.keras.models.load_model("Models/Model "+model_name)
        
    #Wiping the previous directories if you're not restarting training.
    #Makes it a bit easier to restart training
    else:
        dir = data_directory
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        
            
        #Creating the model]
        
        #Adding the input layer
        inputs = tf.keras.Input(shape=(image_size, image_size, 3))
        
        #Getting the architecture
        base_model=tf.keras.applications.resnet_v2.ResNet50V2(include_top=True,input_shape= (image_size, image_size, 3),weights=None,classes=100)(inputs)
        
        #Compiling the model
        Model = tf.keras.Model(inputs, base_model)
                
        Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss= tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='Top 1'),
                               tf.keras.metrics.SparseTopKCategoricalAccuracy(name='Top 5')])
       
        
        #Model.summary()
    
    
    #Loading the dataset. Data is stored as a tensorflow dataset object
    cifar100=tf.data.Dataset.list_files(r'..\..\pre-processing\cifar 100\train\*\*').shuffle(500).map(lambda y: process_image_xy_augment(y,image_size))
    
    cifar100_test=tf.data.Dataset.list_files(r'..\..\pre-processing\cifar 100\test\*\*').shuffle(500).map(lambda y: process_image_xy(y,image_size))
    
    Model.save("Models/Model "+model_name)
    
    #Creating an tensorflow dataset batch object for testing
    batch_cifar100=cifar100.batch(batch_size, drop_remainder=True)
    batch_cifar100_test=cifar100_test.batch(batch_size, drop_remainder=True)
  
    
    #rotation=tf.keras.layers.RandomRotation(1)
    #Main training loop
    for epoch_count in range(0,number_of_epochs):
        
        Model.fit(batch_cifar100,batch_size=batch_size,epochs=1)
        
        history=Model.evaluate(batch_cifar100_test)
        
        #Writing to a file
        with open(data_directory+"/eval.txt", 'a') as f:
            f.writelines([str(epoch_count),",",
                          str(history[1]),",",
                          str(history[2]),",",
                          str(history[0]),"\n"])
            f.close()
            
            
        Model.save("Models/Model "+model_name)
    
    batch_cifar100_test=cifar100_test.batch(2, drop_remainder=True)
    
    history=Model.evaluate(batch_cifar100_test)
    
    with open(data_directory+"/final accuracy.txt", 'a') as f:
        f.writelines([str(history[1]),",",
                      str(history[2]),",",
                      str(history[0]),"\n"])
        f.close()

        
#With Data Augmentation
def process_image_xy_augment(file_path,image_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img,channels=3)
    
    img=tf.image.random_flip_left_right(img)
    img=tf.image.random_flip_up_down(img)
    
    img = tf.image.resize(img, [image_size, image_size])/255
    
    parts = tf.strings.split(file_path, os.path.sep)
    label=tf.strings.to_number(parts[-2])
    return img,label

#Getting images without any data augmentation
def process_image_xy(file_path,image_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.resize(img, [image_size, image_size])/255
    
    parts = tf.strings.split(file_path, os.path.sep)
    label=tf.strings.to_number(parts[-2])
    return img,label

def make_directory(name):
    if(not os.path.isdir(name)):
        os.mkdir(name)
    
if __name__ == '__main__':
    
    #Initializing GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
    random.seed()
    
    main()#from_previous_model,resume_training_epoch,offset)
