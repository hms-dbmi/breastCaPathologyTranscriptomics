
"""
Script by Yasha Ektefaie, November 2020
Used to train models on pathology images (pre-chunked), transfer learning from inceptionv3


"""


import numpy as np
import tables
import pickle
import neptune
from neptunecontrib.monitoring.keras import NeptuneMonitor
import collections
from sklearn.utils import class_weight
import cv2
from scipy import ndimage, misc
import os
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, LSTM, Activation, Masking, Dropout, GlobalAveragePooling2D
import tensorflow as tf
import numpy as np
from keras.optimizers import RMSprop
from keras import backend as k
from sklearn.preprocessing import normalize
from keras.utils import np_utils
from keras import regularizers
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.optimizers import *
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


#Setting up Neptune API Key to track training
neptune_api_key = "Neptune API Key goes here"

neptune.init(api_token = neptune_api_key, project_qualified_name='yashaektefaie/benignmodel')

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    #print(dict(items))
    return dict(items)

#Set up which GPU we use
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#Identify and open up HDF5 file with all data we will use for training
hdf5_path = "/home/ye12/benign_model/benigntumorHDF5.hdf5"

# open the hdf5 file we created
hdf5_file = tables.open_file(hdf5_path, mode='r')

# Open a logger file to store training info
csv_logger = CSVLogger('benign_model_log.tsv', append=True, separator='\t')

# Create the base pre-trained model using inception V3
base_model = InceptionV3(weights='imagenet', include_top=False)

#Adding layers on top of Inception model to train
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.3)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)

#Freeze the main layers of the inceptionv3 so that only the new layers added are trained
for layer in base_model.layers:
    layer.trainable = False

#Initialize Model parameters
lr = 0.001

compile_kwargs = {'loss':tf.keras.losses.BinaryCrossentropy(),
                'optimizer': tf.keras.optimizers.RMSprop(learning_rate=lr),
                'metrics':tf.keras.metrics.BinaryAccuracy()
}

model.compile(**compile_kwargs)


#Batching Function, modified for clarity 
def imageLoader(img, labels, batch_size): 
    batch_size = batch_size
    datasetLength = labels.shape[0]
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < datasetLength:
            limit = min(batch_end, datasetLength)
            X = img[batch_start:limit]
            new_X = []

            for i in X:
                i = cv2.resize(cv2.resize(i, (50, 50)), (224,224))
                new_X.append(preprocess_input(i))

            Y = []
            for i in labels[batch_start:limit]:
                Y.append([np.float32(i)])

            yield (np.array(new_X),np.array(Y))
            batch_start += batch_size   
            batch_end += batch_size

#Set up batch size, experiment for neptune AI
batch_size = 32
train_settings = {'epochs': 25, 'batch_size': batch_size}
neptune_kwargs={}
exp = neptune.create_experiment(params = flatten_dict({'compile_options': compile_kwargs,
                                                      'train_settings': {'learning_rate':lr,
                                                                       **train_settings}}), **neptune_kwargs)

#Set up Checkpoint for model to save best model
checkpoint = ModelCheckpoint("model checkpoint name", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)


#Start training the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
hist = model.fit(
        x=imageLoader(hdf5_file.root.train_img_modified,hdf5_file.root.train_labels_modified,batch_size),
        steps_per_epoch=hdf5_file.root.train_img_modified.shape[0] // batch_size,
        epochs=25,
        validation_data=imageLoader(hdf5_file.root.val_img,hdf5_file.root.val_labels,batch_size),
        validation_steps=hdf5_file.root.val_img.shape[0] // batch_size,
		callbacks=[early_stopping, csv_logger, NeptuneMonitor(exp), checkpoint])

model.save('final model name')
