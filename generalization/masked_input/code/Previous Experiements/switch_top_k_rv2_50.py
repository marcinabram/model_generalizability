import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import os
import numpy as np
import random
from PIL import Image

#This file runs through one epoch for training
def main():
    
    #Model Parameters
    image_size=224
    
    batch_size=64
    
    weight_decay_param=.0001
    
    from_previous_model=0
    
    number_of_epochs=150
    
    
    learning_rate=.05
    
    starting_epoch=0
    
    max_augmnet=.5
    #Directory names
    
    model_name="Top-K Cut Rv2,50 cifar100 switch"
    
    data_directory="Data/"+model_name
    
    image_directory="Images/"+model_name
    
    #Making directories to store stuff
    make_directory(["Models","Data","Images",data_directory,image_directory])

    
    #Loading a previous model
    if(from_previous_model==1):
        
        #Loading the model
        Model_weights=tf.keras.models.load_model("Models/Model "+model_name)
        #Adding the input layer
        inputs = tf.keras.Input(shape=(image_size, image_size, 3))
        #Getting the architecture
        base_model=tf.keras.applications.resnet_v2.ResNet50V2(include_top=True,input_shape= (image_size, image_size, 3),weights=None,classes=100)(inputs)
        
        Model = Masked_Model(inputs,base_model)
        Model.image_size=image_size
        
        Model.set_weights(Model_weights.get_weights())
        for layer in Model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(weight_decay_param)(layer.kernel))
                if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                    layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(weight_decay_param)(layer.bias))
                
                
        Model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=.9),
                      loss= tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='Top 1'),
                               tf.keras.metrics.SparseTopKCategoricalAccuracy(name='Top 5')])
        
        
        Model.set_weights(Model_weights.get_weights())
        
        
    
    #Wiping the previous directories if you're not restarting training.
    #Makes it a bit easier to restart training
    else:
        for directory in [data_directory,image_directory]:
            for f in os.listdir(directory):
                os.remove(os.path.join(directory, f))
            
            
        #Creating the model
        
        #Adding the input layer
        inputs = tf.keras.Input(shape=(image_size, image_size, 3))
        #Getting the architecture
        base_model=tf.keras.applications.resnet_v2.ResNet50V2(include_top=True,input_shape= (image_size, image_size, 3),weights=None,classes=100)(inputs)
        
        #Compiling the model
        Model = Masked_Model(inputs,base_model)
        
        for layer in Model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(weight_decay_param)(layer.kernel))
                if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                    layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(weight_decay_param)(layer.bias))
                
                
        Model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=.9),
                      loss= tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='Top 1'),
                               tf.keras.metrics.SparseTopKCategoricalAccuracy(name='Top 5')])
        
        
    Model.image_size=image_size
    Model.percent=tf.constant(0.05)
    Model.augment_data=tf.constant(0)
    
    
    #Loading the dataset. Data is stored as a tensorflow dataset object
    cifar100=tf.data.Dataset.list_files(r'..\..\pre-processing\cifar 100\train\*\*').shuffle(500).map(lambda y: process_image_xy(y,image_size))
    
    cifar100_test=tf.data.Dataset.list_files(r'..\..\pre-processing\cifar 100\test\*\*').shuffle(500).map(lambda y: process_image_xy(y,image_size))
    
    Model.save("Models/Model "+model_name)
    
    #Creating an tensorflow dataset batch object for testing
    batch_cifar100=cifar100.batch(batch_size, drop_remainder=True)
    batch_cifar100_test=cifar100_test.batch(batch_size, drop_remainder=True)
    
    
    s_data,s_labels=next(iter(batch_cifar100_test))
    sample_data=[]
    sample_labels=[]
    for i in range(16):
        sample_data.append(s_data[i])
        sample_labels.append(s_labels[i])
    sample_data=tf.convert_to_tensor(sample_data)
    sample_labels=tf.convert_to_tensor(sample_labels)
    
    create_image(image_directory, image_size, 0, sample_data,sample_labels, Model)
    
    
    
    previous_accuracy=0
    #Main training loop
    for epoch_count in range(starting_epoch,number_of_epochs):
        if(epoch_count==75):
            tf.keras.backend.set_value(Model.optimizer.lr,learning_rate/10)
            
        if(epoch_count==125):
            tf.keras.backend.set_value(Model.optimizer.lr,learning_rate/100)
            
        if(epoch_count%15==0 and epoch_count!=0):
            Model.augment_data=tf.constant((Model.augment_data.numpy()+1)%2)
            
        print(epoch_count,"   ",Model.percent.numpy())
        
        fit_metrics=Model.fit(batch_cifar100)
        
        
        create_image(image_directory,image_size,epoch_count+1,sample_data,sample_labels,Model)
        
        
            
        history=Model.evaluate(batch_cifar100_test)
        if(Model.augment_data==tf.constant(1)):
            if((fit_metrics.history['Top 1'][0]>=previous_accuracy and fit_metrics.history['Top 1'][0]>.2) or fit_metrics.history['Top 1'][0]>.99):
                Model.percent=tf.constant(Model.percent.numpy()+.1*(max_augmnet-Model.percent.numpy()))
                    
                
            previous_accuracy=fit_metrics.history['Top 1'][0]
        

        
        #Writing to a file
        with open(data_directory+"/eval.txt", 'a') as f:
            f.writelines([str(epoch_count),",",
                          str(history[1]),",",
                          str(history[2]),",",
                          str(history[0]),"\n"])
            f.close()
            
        Model.save("Models/Model "+model_name)
    
    
    batch_cifar100_test=cifar100_test.batch(2, drop_remainder=True)
    
    #Looping through all of the test images. Doing a full evaluation of the model
    
        
    history=Model.evaluate(batch_cifar100_test)
    
    with open(data_directory+"/final accuracy.txt", 'a') as f:
        f.writelines([str(history[1]),",",
                      str(history[2]),",",
                      str(history[0]),"\n"])
        f.close()

class Masked_Model(tf.keras.Model):
    
    image_size=64
    loss=tf.keras.losses.SparseCategoricalCrossentropy()
    flipping=tf.keras.layers.RandomFlip()
    percent=tf.constant(0)
    augment_data=tf.constant(0)
    
    def train_step(self, next_batch):
        Data,Labels=next_batch
        
        Data=self.flipping(Data)
        Data=self.flipping(Data)
        def identity(): return Data
        Data=tf.cond(self.augment_data==tf.constant(0),identity,lambda: mask_data(self.image_size,Data,Labels,self,self.percent))
        
        with tf.GradientTape() as tape:
            classification_vector=self(Data)
        
            loss=self.compiled_loss(Labels,classification_vector, regularization_losses=self.losses)
            #loss+=sum(self.losses)
        gradients = tape.gradient(loss, self.trainable_weights)
        #print(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(Labels, classification_vector)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    
    
def create_image(image_directory,image_size,epoch_count,sample_data,sample_labels,Model):
    image_array_original=[]
    image_array_masked=[]
    
    #Creating the mask
    data_test_mask=mask_data(image_size,sample_data,sample_labels,Model,Model.percent)
    
    #saving the images
    for i in range(4):
        image_list1=[]
        image_list2=[]
        
        for j in range(4):
            image_list1.append(Image.fromarray(np.uint8(sample_data[i*4+j]*255)))
            image_list2.append(Image.fromarray(np.uint8(data_test_mask[i*4+j]*255)))
            
        image_array_original.append(image_list1)
        image_array_masked.append(image_list2)
    
    full_image=Image.new('RGB', (image_size*2*2, image_size*2*2))
    
    for i in range(4):
        for j in range(4*2):
            if(j%2==0):
                full_image.paste(image_array_original[i][int(j/2)], (j*image_size,i*image_size))
            if(j%2==1):
                full_image.paste(image_array_masked[i][int(j/2)], (j*image_size,i*image_size))
    full_image.save(image_directory+"/"+str(epoch_count)+".jpeg")
    

def mask_data(image_size,Data,Labels,Model,percent):
    
   
    
    #Calculating the gradients. Note this can also be used to do a basic
    #adversarial attack on the model for training easily 
    #I'm omitting this code to test the validity of my technique 
    #in isolation. Will test at a later date
    #Just need to add the line data+=alpha*gradients (alpha is a constant)
    k=tf.cast(tf.math.floor(percent*3*image_size*image_size),dtype=tf.int32)
    
    with tf.GradientTape() as tape_image:
        tape_image.watch(Data)
        classification_vector=Model(Data,training=False)
        
        Vectors=tf.gather_nd(classification_vector,Labels)
        
        
    #Actually getting the gradient
    gradients=tape_image.gradient(Vectors, Data)
    
    #gradients=gaussian_filter(gradients, sigma=1)
    
    #flattening the image makes it easier to find the max
    line_grad = tf.math.abs(tf.keras.layers.Flatten()(gradients))
    
    Data=tf.keras.layers.Flatten()(Data)
    
    Data_shuffled=tf.random.shuffle(Data)
    
    max_index=tf.expand_dims(tf.math.top_k(line_grad,k=k).indices,axis=2)
    
    
    
    
    batch=tf.expand_dims(tf.cast(tf.linspace(tf.zeros(k),(max_index.shape[0]-1)*tf.ones(k),max_index.shape[0],axis=0),dtype=tf.int32),axis=2)
    
    
    indices=tf.concat([batch,max_index],axis=2)
    indices=tf.reshape(indices,shape=[indices.shape[0]*indices.shape[1],indices.shape[2]])
    
    updates=tf.gather_nd(Data_shuffled,indices)
    
    
    
    Data=tf.tensor_scatter_nd_update(Data, indices, updates)
    
    Data=tf.keras.layers.Reshape((image_size,image_size,3))(Data)

    return Data



def make_directory(names):
    for name in names:
        if(not os.path.isdir(name)):
            os.mkdir(name)
            
#Getting images and labels
def process_image_xy(file_path,image_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.resize(img, [image_size, image_size])/255
    
    parts = tf.strings.split(file_path, os.path.sep)
    label=tf.strings.to_number(parts[-2])
    return img,label

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
