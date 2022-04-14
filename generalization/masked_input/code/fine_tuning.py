import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import tensorflow_addons as tfa

#This file runs through one epoch for training
def main():
    
    #Model Parameters
    image_size=224
    
    batch_size=64
    
    weight_decay_param=.0001
    
    number_of_epochs=10
    
    #Number of regions used in numerical intergration
    number_of_steps=tf.constant(10)
    
    learning_rate=3*(10**(-4))
    
    from_previous_model=0
    #Directory names
    
    model_name="Percent Attribution Rv2,50 cifar100 fine_tuning"
    
    previous_model_name="Base Model Rv2,50 cifar100"
    
    data_directory="../Data/"+model_name
    
    
    
    #Making directories to store stuff
    make_directory("../Models")
    
    make_directory("../Data")
    
    make_directory(data_directory)
    
    #Loading a previous model
    if(from_previous_model==0):
        dir = data_directory
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        
            
    #Creating the model]
    Model_Weights=tf.keras.models.load_model("Models/Model "+previous_model_name)
    
    #Adding the input layer
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    
    #Getting the architecture
    base_model=tf.keras.applications.resnet_v2.ResNet50V2(include_top=True,input_shape= (image_size, image_size, 3),
                                                          weights=None,classes=100)(inputs)
    
    #Compiling the model
    Model = Masked_Model(inputs, base_model)
            
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss= tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='Top 1'),
                           tf.keras.metrics.SparseTopKCategoricalAccuracy(name='Top 5')])
    
    Model.set_weights(Model_Weights.get_weights())
    
    #Initializing model params
    Model.percent=tf.convert_to_tensor([.3])
    Model.Steps=number_of_steps
    Model.image_size=224
    
    

    
    
    #Loading the dataset. Data is stored as a tensorflow dataset object
    cifar100=tf.data.Dataset.list_files(r'..\..\pre-processing\cifar 100\train\*\*'
                                        ).shuffle(500).map(lambda y: process_image_xy_augment(y,image_size))
    
    cifar100_test=tf.data.Dataset.list_files(r'..\..\pre-processing\cifar 100\test\*\*'
                                             ).shuffle(500).map(lambda y: process_image_xy(y,image_size))
    
    Model.save("Models/Model "+model_name)
    
    #Creating an tensorflow dataset batch object for testing
    batch_cifar100=cifar100.batch(batch_size, drop_remainder=True)
    batch_cifar100_test=cifar100_test.batch(batch_size, drop_remainder=True)
  
    
    #rotation=tf.keras.layers.RandomRotation(1)
    #Main training loop
    
    #Model.evaluate(batch_cifar100_test)
    for epoch_count in range(0,number_of_epochs):
        
        #if(epoch_count>(200+40)):
         #   Model.optimizer.lr.assign(.0025)
        #Looping through all of the batches
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
    percent=tf.convert_to_tensor([0])
    Steps=tf.constant(10)
    
    def train_step(self, next_batch):
        Data,Blurred,Labels=next_batch
        #Creating the saliency map
        saliency_map=tf.map_fn(lambda X: self.saliency_map(X),(Data, Blurred, Labels),fn_output_signature=tf.float32)
        
        #Applying the mask (or cut in this case)
        Data=self.Mask_Data(Data,saliency_map)
        
        #Training the model
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
    
    @tf.function
    #This creates the saliency map for a given image from the intergrated
    #gradients method
    def saliency_map(self,X):
        Data,Baseline,Labels=X
        Data=tf.expand_dims(Data,axis=0)
        Baseline=tf.expand_dims(Baseline,axis=0)
        difference_step=(Data-Baseline)
        
        #Creating the datapoints to estimate the intergral by
        alphas = tf.linspace(start=0.0, stop=1.0, num=self.Steps+1)
        
        alphas = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        
        path_inputs = Baseline +  alphas * difference_step
        
        
            
        #Getting the classification outputs on the datapoints
        with tf.GradientTape() as tape_image:
            tape_image.watch(path_inputs)
            classification_vector=self(path_inputs,training=False)
            classification_vector=tf.slice(classification_vector,[0, tf.cast(Labels,tf.int32)],[self.Steps+1,1])
            
            
        #Actually getting the gradient with respect to the target class
        
        grads=tape_image.gradient(classification_vector, path_inputs)
        
        #Getting the intergral values via the trapezoidal rule
        grads=(grads[:-1] + grads[1:]) / tf.constant(2.0)
        grads= tf.math.reduce_mean(grads, axis=0)
        
        #Adding in the normalizing factor (X-X')
        saliency_map=difference_step*grads
        return tf.math.abs(saliency_map)[0]
    
    @tf.function
    #This applies the mask given the Data and Saliency Mp
    def Mask_Data(self,Data,saliency_map):
        '''The mask is applied to cover a certain % of the contribution. So a
        30% mask doesn't mask 30% of the pixels. So for instance given 2 vectors
        {.1,.2,.4,...,1} and {.25,.4,...,1} 2 items would be masked in the first
        while only one would be in the second'''
        
        #gradients=gaussian_filter(gradients, sigma=1)
        
        #flattening the image makes it easier to find the max
        line_map = tf.keras.layers.Flatten()(saliency_map)
        #Normalizing everything
        
        #Need to know the total amount of conribution in each image
        reduced_sum=tf.math.reduce_sum(line_map,axis=1)
        
        #A bit easier to isntead of normalizing everything to 1, instead figure
        #out how much of the toal contribution needs to be masked.
        percent_map= tf.expand_dims(self.percent*reduced_sum,axis=1)
        
        #Sorting the map in increasing order. This is needed because the largest
        #item needs to be masked first, then second, then third etc...
        
        #This creates the arguments from least to greatest which would result
        #in a sorted list. Because the list needs to be from greatest to least
        #it has to be multiplied by -1
        sorted_saliency_order=tf.argsort(-1*line_map)
        
        #Actually creating the sorted list
        sorted_saliency=tf.gather(line_map,sorted_saliency_order, batch_dims=-1)
        
        #Need the cumalitive sum to figure out where the cutoff is
        cum_sum=tf.math.cumsum(sorted_saliency,axis=1)
        
        Data=tf.keras.layers.Flatten()(Data)
        
        #Applying a cut based on random items in the same batch
        Data_shuffled=tf.random.shuffle(Data)
        
        #The number of pixels to mask for each image
        amount_to_mask=tf.searchsorted(cum_sum,percent_map)
        
        #Actually creating the cut and putting it onto the image
        Data=tf.map_fn(lambda X: self.Apply_Mask(X),(Data,amount_to_mask,sorted_saliency_order,Data_shuffled,percent_map),fn_output_signature=tf.float32)

        return Data
    
    #This function creates and applies a mask
    @tf.function
    def Apply_Mask(self,X):
        Data,amount_to_mask,sorted_saliency_order,Data_shuffled,percent_map=X
        
        #Getting the indices needed to be masked
        indices=tf.slice(sorted_saliency_order,[0],amount_to_mask)
        indices=tf.expand_dims(indices,axis=1)
        
        #Getting the updates to be put in the original image
        updates=tf.gather_nd(Data_shuffled,indices)
        
        #Applying the updates to the original image
        Data_Masked=tf.tensor_scatter_nd_update(Data, indices, updates)
        Data_Masked=tf.reshape(Data_Masked,[self.image_size,self.image_size,3])
        return Data_Masked
        
        
    

def process_image_xy_augment(file_path,image_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img,channels=3)
    
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)/255
    
    blurred_image=tfa.image.gaussian_filter2d(img,12,16)
    
    img = tf.image.resize(img, [image_size, image_size])
    blurred_image=tf.image.resize(blurred_image, [image_size, image_size])
    
    parts = tf.strings.split(file_path, os.path.sep)
    label=tf.strings.to_number(parts[-2])
    return img,blurred_image,label
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
        
    
    main()#from_previous_model,resume_training_epoch,offset)
