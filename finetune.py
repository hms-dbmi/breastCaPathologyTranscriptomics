
"""
Script by Yasha Ektefaie, November 2020
Used to finetune models trained on pathology images (pre-chunked) on External Images

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
from tensorflow.keras.models import load_model
import os
import random
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
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


#Set up GPU to use for fine tuning
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#Set up Neptune API to use for tracking training progress
neptune_api_key = "Neptune API Key"

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


#Identify and open HDF5 file containing data to finetune with
hdf5_path = "/home/ye12/benign_model/benigntumorHDF5.hdf5"

hdf5_file = tables.open_file(hdf5_path, mode='r')


csv_logger = CSVLogger('benign_model_log.tsv', append=True, separator='\t')

# Identify and open model to fine tune
path_to_model = "model to finetune"
model = load_model(path_to_model, compile=True)

#Unfreeze specific layers of model to finetune
for layer in model.layers[249:]:
    layer.trainable=True

# Recompile model with low learning rate and SGD optimizer foor finetuning
lr = 0.00001
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss="binary_crossentropy",metrics=['accuracy'])

#Regular Batch Generating Function, used to load data unmodified
def imageLoader(img, labels, batch_size, validation=0): 
    datasetLength = labels.shape[0]
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < datasetLength:
            limit = min(batch_end, datasetLength)
            X = img[batch_start:limit]
            if validation:
                new_X = []
                for i in X:
                    i = cv2.resize(cv2.resize(i, (50, 50)), (224,224))
                    new_X.append(preprocess_input(i))
                Y = np.array([[np.float32(i)] for i in labels[batch_start:limit]])

                yield (np.array(new_X),Y)
            else:
                yield (X, labels[batch_start:limit])
            batch_start += batch_size   
            batch_end += batch_size

#Modified Batch Generating Function, used to load a proportion of External Data interspersed with training data to protect against catastrophic forgetting
def imageLoader_modified(batch_size):
    datasetLength = hdf5_file.root.exttest_labels.shape[0]*0.40
    swap = 0
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < datasetLength:
            limit = min(batch_end, datasetLength)
            if swap%5 == 0:
                X = hdf5_file.root.train_img_modified[batch_start:limit]
                Y = hdf5_file.root.train_labels_modified[batch_start:limit]
            else:
                if swap%2 == 0:
                    X = hdf5_file.root.exttest_img[batch_start:limit]
                    Y = hdf5_file.root.exttest_labels[batch_start:limit]
                else:
                    X = hdf5_file.root.exttest_img[batch_start+100000:limit+100000]
                    Y = hdf5_file.root.exttest_labels[batch_start+100000:limit+100000]
            new_X = []
            for i in X:
                i = cv2.resize(cv2.resize(i, (50, 50)), (224,224))
                new_X.append(preprocess_input(i))
            yield (np.array(new_X),Y)
            swap += 1

            batch_start += batch_size
            batch_end += batch_size

#Batch Size and Neptune Experiment Setup
batch_size = 32
train_settings = {'epochs': 50, 'batch_size': batch_size}
neptune_kwargs={}
exp = neptune.create_experiment(params = flatten_dict({'compile_options': compile_kwargs,
                                                       'train_settings': {'learning_rate':lr,
                                                                        **train_settings}}), **neptune_kwargs)

#Setup Checkpoint to save best model iteration
checkpoint = ModelCheckpoint("Model Checkpoint Name", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

#Start FineTuning The Model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
hist = model.fit(
	x=imageLoader_modified(batch_size),
	steps_per_epoch=hdf5_file.root.exttest_img.shape[0]*0.4 // batch_size,
        epochs=50,
        validation_data=imageLoader(hdf5_file.root.val_img,hdf5_file.root.val_labels,batch_size, 1),
        validation_steps=hdf5_file.root.val_img.shape[0] // batch_size,
	callbacks=[early_stopping, csv_logger, NeptuneMonitor(exp), checkpoint])

model.save('Final Model Name')
