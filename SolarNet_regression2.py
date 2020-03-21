#Dipole Prediction
#By: Bernard Billy Jason Vattepu Benson 


from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input
from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout
from keras.optimizers import SGD
from keras.models import model_from_json

path = '/home/exx/data/ar11890_data/train/ar11890_c_AIA94_0000_al_0.0108603_t_2013.11.08_01:36:00_TAI.png'
label = 1
csv_path = '/home/exx/data/ar11890_data/train.csv'

def get_sample(path,label):
    
    img = cv2.imread(path)
    
    
    return(img, label)
    
#    
def preprocess(img):
    
    normalized_img = img/255.0
    
    return normalized_img

def imagegenerator(csv_path,batch_size=128):
    
   alphas = pd.read_csv(csv_path, header=None)
   ind = range(0,len(alphas))
   while True:
       batch_ind = np.random.choice(ind, size=batch_size)
       
       images=[]
       labels=[]
       for index in batch_ind:
           img, label = get_sample(alphas.iloc[index,0],alphas.iloc[index,1])
           images.append(img)
           labels.append(label)
       images=np.asarray(images)
       labels=np.asarray(labels)
       yield images,labels
       
def test_imagegenerator(csv_path,batch_size=1347):
    
   alphas = pd.read_csv(csv_path, header=None)
   ind = range(0,len(alphas))

   batch_ind = np.random.choice(ind, size=batch_size)
   batch_ind = np.arange(batch_size)      
   images=[]
   labels=[]
   for index in batch_ind:
        img, label = get_sample(alphas.iloc[index,0],alphas.iloc[index,1])
        images.append(img)
        labels.append(label)
   images=np.asarray(images)
   labels=np.asarray(labels)
   return images, labels

       
img_width, img_height = 256,256
#train_data_dir = '/data/home/bb0008/Documents/Images/Sixteen_Dipoles/train'
#validation_data_dir = '/data/home/bb0008/Documents/Images/Sixteen_Dipoles/val'
#test_data_dir = '/data/home/bb0008/Documents/Images/Sixteen_Dipoles/test'
nb_train_samples = 6288
nb_validation_samples = 1349
epochs = 100
batch_size = 128
batches_per_epoch_train = int(nb_train_samples/batch_size)
batches_per_epoch_test = int(nb_validation_samples/batch_size)
nb_test_samples = 1347

input_tensor = Input(shape=(img_height, img_width, 3))
with tf.device('/cpu:0'):
   model = Sequential()
   model.add(Conv2D(32, (5,5), activation='relu', input_shape=(256,256,3)))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3))
   model.add(Activation('relu'))
   model.add(MaxPooling2D((2, 2)))
   model.add(Dropout(0.25))
   
   model.add(Conv2D(64, (3, 3), activation='relu'))
   model.add(Conv2D(64, (1, 1), activation='relu'))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3))
   model.add(Activation('relu'))
   model.add(MaxPooling2D((2, 2)))
   model.add(Dropout(0.25))
   
   model.add(Conv2D(64, (3, 3), activation='relu'))
   model.add(Conv2D(64, (1, 1), activation='relu'))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3))
   model.add(Activation('relu'))
   model.add(MaxPooling2D((2, 2)))
  
   
   model.add(Conv2D(128, (3, 3), activation='relu'))
   model.add(Conv2D(128, (1, 1), activation='relu'))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3))
   model.add(Activation('relu'))
   model.add(MaxPooling2D((2, 2)))
   
   model.add(Conv2D(256, (3, 3), activation='relu'))
   model.add(Conv2D(256, (1, 1), activation='relu'))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3))
   model.add(Activation('relu'))
   #model.add(MaxPooling2D((2, 2)))
   
   model.add(Conv2D(256, (3, 3), activation='relu'))
   model.add(Conv2D(256, (1, 1), activation='relu'))
   model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3))
   model.add(Activation('relu'))
   model.add(MaxPooling2D((2, 2)))
   model.add(Flatten())
   model.add(Dropout(0.25))
   

   model.add(Dense(1024, activation='relu'))
   model.add(Dense(1024, activation='relu'))
   model.add(Dropout(0.5))
   model.add(Dense(1, activation='linear'))

parallel_model = multi_gpu_model(model, gpus=4)

parallel_model.compile(loss="logcosh",
              optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6))
parallel_model.summary()



#Train the model
train_path = '/home/exx/data/ar11890_data/train.csv'
val_path = '/home/exx/data/ar11890_data/val.csv'
test_path = '/home/exx/data/ar11890_data/test.csv'

my_model = parallel_model.fit_generator(
    generator=imagegenerator(train_path),
    steps_per_epoch=batches_per_epoch_train,
    epochs=epochs,
    validation_data=imagegenerator(val_path),
validation_steps=batches_per_epoch_test)



# serialize model to JSON
model_json = model.to_json()
with open("/home/exx/Documents/Bernard/Sharp_params_prediction/Models/SolarNet_32batch_200epochs.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/exx/Documents/Bernard/Sharp_params_prediction/Models/SolarNet_32batch_200epochs.h5")
print("Saved model to disk")

x,y = test_imagegenerator(test_path)
#predictions = parallel_model.predict(imagegenerator(test_path),steps = batches_per_epoch_test, verbose=1)
predict_np = parallel_model.predict(x, steps = 1, verbose=1)

import matplotlib.pyplot as plt
#plt.plot(my_model.history['acc'])
#plt.plot(my_model.history['val_acc'])
#plt.title('ResNet50_Sixteen_Dipoles_Accuracy_128batch')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='lower right')
#plt.show()
#
plt.plot(my_model.history['loss'])
plt.plot(my_model.history['val_loss'])
plt.title('SolarNet_AR11283_Loss_64batch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower middle')
plt.show()

plt.plot(y,predict_np)
plt.show()
#for i in range(len(predict_np)):
#	print("Predicted=%s, Actual= %s" % (predict_np[i], y[i]))

#from keras.models import model_from_json
#from keras.layers.core import Dense, Activation
#
## Loading and cutting
#parallel_model_new = model_from_json(open('/home/exx/Documents/Bernard/Sharp_params_prediction/Models/SolarNet_32batch_200epochs.json').read())
#parallel_model_new.load_weights('/home/exx/Documents/Bernard/Sharp_params_prediction/Models/SolarNet_32batch_200epochs.h5')
##parallel_model.layers = parallel_model.layers[0:-2]
#
#parallel_model_new.summary()
#predictions = parallel_model_new.predict_generator(imagegenerator(test_path),steps = 1, verbose = 1)
#predict_np = parallel_model_new.predict(x, steps = 1, verbose = 1)



#test_images, labels = imagegenerator(test_path,batch_size = 8250)
#filenames = test_generator.filenames
#nb_samples = len(filenames)
#predictions = parallel_model.predict_generator(test_generator,steps =None, verbose = 1)
#import numpy as np
#prediction_np = np.argmax(predictions, axis = -1)
##
##Writing csv files to output
#import csv
#csvfile = "/data/home/bb0008/Documents/Python/Revision_Results/Regression/MSE_labels.csv"
#with open(csvfile,"w") as output:
#     writer = csv.writer(output, lineterminator = '\n')
#     for val in filenames:
#         writer.writerow([val])
#csvfile = "/data/home/bb0008/Documents/Python/Revision_Results/Regression/MSE_predictions.csv"
#with open(csvfile,"w") as output:
#     writer = csv.writer(output, lineterminator = '\n')
#     for val in prediction_np:
#        writer.writerow([val])

my_model.history




































