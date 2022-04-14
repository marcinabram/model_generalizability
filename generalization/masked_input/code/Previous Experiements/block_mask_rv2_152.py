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
    image_size=64
    
    batch_size=128
    
    from_previous_model=0
    
    number_of_epochs=50
    
    #Directory names
    
    model_name="Block Masking Rv2,152"
    
    data_directory="Data/"+model_name
    
    image_directory="Images/"+model_name
    
    
    #Making directories to store stuff
    if(not os.path.isdir("Models")):
        os.mkdir("Models")
    
    if(not os.path.isdir("Data")):
        os.mkdir("Data")
        
    if(not os.path.isdir("Images")):
        os.mkdir("Images")
    
    if(not os.path.isdir(data_directory)):
        os.mkdir(data_directory)
        
    if(not os.path.isdir(image_directory)):
        os.mkdir(image_directory)
    
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
            
        dir = image_directory
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
            
        #Creating the model]
        
        
        #Getting the architecture
        Model=tf.keras.applications.resnet_v2.ResNet152V2(include_top=False,input_shape= (image_size, image_size, 3),weights=None,classes=100)
        
        
        
        Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                         loss= tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='Top 1'),
                                  tf.keras.metrics.SparseTopKCategoricalAccuracy(name='Top 5')])
    
    
    #Loading the dataset. Data is stored as a tensorflow dataset object
    cifar100=tf.data.Dataset.list_files(r'..\..\pre-processing\cifar 100\train\*\*').shuffle(500).map(lambda y: process_image_xy_r(y,image_size))
    
    cifar100_test=tf.data.Dataset.list_files(r'..\..\pre-processing\cifar 100\test\*\*').shuffle(500).map(lambda y: process_image_xy(y,image_size)).repeat()
    
    
    #Creating an tensorflow dataset batch object for testing
    batch_cifar100_test=cifar100_test.batch(batch_size, drop_remainder=True)
    
    #Getting images to display
    batch_cifar100_example=cifar100_test.batch(16, drop_remainder=True)
    
    cifar100_next_batch=next(iter(batch_cifar100_example))
    data_test,Labels=cifar100_next_batch
    
    #Keeping track of the number of training steps
    step=0
    
    
    #Main training loop
    for epoch_count in range(number_of_epochs):
        
        batch_cifar100=cifar100.batch(batch_size, drop_remainder=True)
        
        
        #Looping through all of the batches
        for image in batch_cifar100:
            
            #if we're restarting only train on the images we haven't seen in an epoch
            with tf.device('/GPU:0'):
                
                #Every Epoch save some example images from the test set
                if(step%390==0):
                    create_image(image_directory,image_size,epoch_count,data_test,Model)
                
                
                
                #Seperating the data and the labels
                data,Labels=image
                    
                #Need to run the training algorithim with out masking for a couple epochs
                #Not entirely sure why, but if that doesn't happen training slows down
                #significantly and converges to the equivlent of not using it at all
                if(epoch_count>=3):
                    #Creating the mask
                    data=mask_data(image_size,data,Model)
                
                
                
                #fitting the model
                history=Model.fit(data,Labels,batch_size=batch_size,epochs=1,verbose=0)
                
                #Writing the results to a file
                with open((data_directory+"/training data.txt"), 'a') as f:
                    f.writelines([str(step),',',
                                  str(history.history['Top 1'][0]),",",
                                  str(history.history['Top 5'][0]),",",
                                  str(history.history['loss'][0]),"\n"])
                    f.close()
            
                
                
                #periodically saving and evaluating the model
                if((step+1)%195==0 and step!=0):
                    Model.save("Models/Model "+model_name)
                    
                    #The metrics for the model are returned in a vector
                    #containg [Loss Value, Top-1 Accuracy, Top-5 Accuracy]
                    metrics=np.zeros(3)
                    
                    for k in range(20):
                        cifar100_next_batch=next(iter(batch_cifar100_test))
                        data,Labels=cifar100_next_batch
                        
                        history=Model.test_on_batch(data,Labels)
                        metrics+=history
                    
                    #calculating the average
                    metrics/=20
                    
                    print("Step: ",str(step+1),"  ",
                          "Model Evaluation Loss: ",str(metrics[0]),
                          " Model Evaluation Accuracy: ",str(metrics[1]),
                          " Top 5: ",str(metrics[2]))
                    
                    #Writing to a file
                    with open(data_directory+"/eval.txt", 'a') as f:
                        f.writelines([str(step+1),",",
                                      str(metrics[1]),",",
                                      str(metrics[2]),",",
                                      str(metrics[0]),"\n"])
                        f.close()
                #Keeping count of the number of steps in the epoch
                step+=1
                    
            
            
        Model.save("Models/Model "+model_name)
        
    
    create_image(image_directory,image_size,(epoch_count+1),data_test,Model)
    
    cifar100_test=tf.data.Dataset.list_files(r'..\pre-processing\cifar 100\test\*\*').shuffle(500).map(lambda y: process_image_xy(y,image_size))
    batch_cifar100=cifar100_test.batch(2, drop_remainder=True)
    
    #Looping through all of the test images. Doing a full evaluation of the model
    number=0
    metrics=np.zeros(3)
    for image in batch_cifar100:
        number+=1
        cifar100_next_batch=next(iter(batch_cifar100_test))
        data,Labels=cifar100_next_batch
        
        history=Model.test_on_batch(data,Labels)
        
        metrics+=history
    metrics/=number
    
    with open(data_directory+"/final accuracy.txt", 'a') as f:
        f.writelines([str(metrics[1]),",",
                      str(metrics[2]),",",
                      str(metrics[0]),"\n"])
        f.close()

def create_image(image_directory,image_size,epoch_count,data_test,Model):
    
    image_array_original=[]
    for i in range(4):
        image_list=[]
        for j in range(4):
            image_list.append(Image.fromarray(np.uint8(data_test[i*4+j]*255)))
        image_array_original.append(image_list)
    
    #Creating the mask
    data_test_mask=mask_data(image_size,data_test,Model)
    
    #saving the masked images
    image_array_masked=[]
    for i in range(4):
        image_list=[]
        for j in range(4):
            image_list.append(Image.fromarray(np.uint8(data_test_mask[i*4+j]*255)))
        image_array_masked.append(image_list)
    
    
    full_image=Image.new('RGB', (image_size*2*2, image_size*2*2))
    
    for i in range(4):
        for j in range(4*2):
            if(j%2==0):
                full_image.paste(image_array_original[i][int(j/2)], (j*image_size,i*image_size))
            if(j%2==1):
                full_image.paste(image_array_masked[i][int(j/2)], (j*image_size,i*image_size))
    full_image.save(image_directory+"/"+str(epoch_count)+".jpeg")
            
    
#This function calculates and applies the gradient based masking to the images
def mask_data(image_size,data,Model):
    
    #Calculating the gradients. Note this can also be used to do a basic
    #adversarial attack on the model for training easily 
    #I'm omitting this code to test the validity of my technique 
    #in isolation. Will test at a later date
    #Just need to add the line data+=alpha*gradients (alpha is a constant)
    with tf.GradientTape() as tape_image:
        tape_image.watch(data)
        classification_vector=Model(data,training=False)
        
    #Actually getting the gradient
    gradients=tape_image.gradient(classification_vector, data)
    
    #flattening the image makes it easier to find the max
    line_image=tf.keras.layers.Flatten()(gradients)
    
    #Getting the max (absolute) value index
    max_index=tf.math.argmax(tf.math.abs(line_image),axis=1)
    
    #This is where we actually apply the mask
    new_data=tf.Variable(data)
    
    #The mask is applied to n pixels out from the center
    #(so an 2n+1 by 2n+1 box) across all 3 color channels
    #If the image size is 64 n=1
    
    #For every image in the batch
    for i in range(line_image.shape[0]):
        
        #Converting the index gotten into an row,column index for the ith image
        
        row=tf.cast(max_index[i]%image_size,dtype=tf.int32)
        column=tf.cast((max_index[i]%(image_size**2))/image_size,dtype=tf.int32)
        
        #for every index in the row to be masked
        for j in range(-int(.025*image_size),int(.025*image_size)+1):
            
            #for every index in the column to be masked
            for k in range(-int(.025*image_size),int(.025*image_size)+1):
                
                #for each channel
                for channel in range(3):
                    new_data[i,(row+j)%image_size,(column+k)%image_size,channel].assign(0)
                
    #Converting back into a tensor for training
    return tf.convert_to_tensor(new_data)

#Getting images with data augmentation
def process_image_xy_r(file_path,image_size):
    
    #Loading the image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.resize(img, [image_size, image_size])/255
    
    #Applying augmentations
    img=tf.image.rot90(img,random.randint(0, 3))
    img=tf.image.random_flip_left_right(img)
    img=tf.image.random_flip_up_down(img)
    
    #Getting Label
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
