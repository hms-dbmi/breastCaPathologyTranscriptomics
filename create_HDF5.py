
"""
Script by Yasha Ektefaie, November 2020
Example of script used to create the HDF5 file used to store data


"""

import pandas as pd
import numpy as np
import tables
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import random
from tqdm import tqdm


#Extract Metadata for the TCGA dataset to map patient barcode to cancer type

#Identify and Open MetaData for TCGA data
subtype_data = "/mnt/data2/datasets/breastCa/TCGA_path/Clinical/nationwidechildrens.org_clinical_patient_brca.txt"
TCGA_metadata = pd.read_csv(subtype_data,sep='\t')

histological_values_TCGA = TCGA_metadata[['bcr_patient_barcode','histological_type']].set_index('bcr_patient_barcode').T.to_dict('list')

label_to_histology_TCGA = {'Infiltrating Ductal Carcinoma':'ductal',
'Infiltrating Lobular Carcinoma':'lobular'}

final_TCGA_histological_labels = {}

for key in histological_values_TCGA:
    value = histological_values_TCGA[key]
    if 'Infiltrating Ductal Carcinoma' in value:
        final_TCGA_histological_labels[key] = 'ductal'
    elif 'Infiltrating Lobular Carcinoma' in value:
        final_TCGA_histological_labels[key] = 'lobular'
    else:
        final_TCGA_histological_labels[key] = 'none'

#Identify and Extract Metadata on whether images in TCGA are cancerous or not based on previous modl
cancerous_images = []
for line in open("metadata_TCGA"):
    data = line.split("\t")
    if int(data[1]):
        cancerous_images.append(data[0])

#Now we go through all images and aggregate them with metadata patient level
data_location = "/mnt/data2/datasets/breastCa/TCGA_path/256dense200"

TCGA_lobular_images = [] 
TCGA_ductal_images = []
patients_present = {}

directories_to_search = [i for i in os.listdir(data_location) if 'TCGA-' in i]

for directory in tqdm(directories_to_search, total=len(directories_to_search)):
    found_label = None
    
    for i in final_TCGA_histological_labels:
        if i in directory:
            found_label = i
    
    if not found_label:
        print("Was not able to find {}".format(directory))
    else:

        if found_label not in patients_present:
            patients_present[found_label] = []

        value = final_TCGA_histological_labels[found_label]
        if value != 'none':
            image_to_create = [i for i in os.listdir(data_location+'/'+directory) if ".jpg" in i]
            for image in image_to_create:
                image_path = data_location+'/'+directory+'/'+image
                if image_path in cancerous_images:
                    if value == 'lobular':
                        TCGA_lobular_images.append(image_path)
                        patients_present[found_label].append([image_path,0])
                    elif value == 'ductal':
                        TCGA_ductal_images.append(image_path)
                        patients_present[found_label].append([image_path,1])                     

print("Finished processing {} lobular images and {} ductal images".format(len(TCGA_lobular_images), len(TCGA_ductal_images)))

#Split data based on patients 80% train, 10% test, 10% validationo
patients_present_keys = list(set(patients_present.keys()))
random.shuffle(patients_present_keys)

train_patients_cutoff = int(len(patients_present_keys)*0.8)
test_patients_cutoff = int(len(patients_present_keys)*0.1)

train_patients = patients_present_keys[:train_patients_cutoff]
test_patients = patients_present_keys[train_patients_cutoff:train_patients_cutoff+test_patients_cutoff]
val_patients = patients_present_keys[train_patients_cutoff+test_patients_cutoff:]

train_x = []
train_y = []

train_x_pos = []
train_x_neg = []

train_x_modified = []
train_y_modified = []

test_x = []
test_y = []

val_x = []
val_y = []


for i in train_patients:
    for data_point in patients_present[i]:
        point, value = data_point[0], data_point[1]
        train_x.append(point)
        train_y.append(value)
        if value == 1:
            train_x_pos.append(point)
        else:
            train_x_neg.append(point) 

for i in train_x_pos:
    train_x_modified.append(i)
    train_y_modified.append(1)

for j in range(int(len(train_x_pos)/len(train_x_neg))):
    for i in train_x_neg:
        train_x_modified.append(i)
        train_y_modified.append(0)

for i in test_patients:
    for data_point in patients_present[i]:
        point, value = data_point[0], data_point[1]
        test_x.append(point)
        test_y.append(value)

for i in val_patients:
    for data_point in patients_present[i]:
        point, value = data_point[0], data_point[1]
        val_x.append(point)
        val_y.append(value) 

print("Length of train_X {}".format(len(train_x)))
print("Length of test_X {}".format(len(test_x)))
print("Length of val_X {}".format(len(val_x)))


#Now we do similar for Sunnybrook data, extracting Metadata
excel = pd.read_excel("/mnt/data2/datasets/breastCa/Sunnybrook/Clinical_Features.xlsx")

columns_interest = [i for i in excel.columns if 'scan ID' in i or 'Histological' in i]
print(columns_interest)
excel_interest = excel[columns_interest].copy()
histological_values = excel_interest.set_index('scan ID ').T.to_dict('list')

final_histological_labels = {}

for key in histological_values:
    value = histological_values[key]
    if 'IDC' in value:
        final_histological_labels[key] = 'ductal'
    elif 'ILC' in value:
        final_histological_labels[key] = 'lobular'
    else:
        final_histological_labels[key] = 'none'
    

data_location = "/mnt/data2/datasets/breastCa/Sunnybrook/tiled/256dense200"
alternative_data_location = "/mnt/data2/ye12/chunk/completed"

lobular_images = [] 
ductal_images = []

already_scanned = []

for directory in os.listdir(alternative_data_location):
    value = final_histological_labels[int(directory)]
    if value != 'none':
        for image in os.listdir(alternative_data_location+'/'+directory):
            if value == 'lobular':
                already_scanned.append(directory)
                lobular_images.append(alternative_data_location+'/'+directory+'/'+image)
            else:
                raise Exception("Not lobular for somem reason {}\t{}".format(directory, value))


for directory in os.listdir(data_location):
    if "test" not in directory and "Stats" not in directory and int(directory) in final_histological_labels and directory not in already_scanned:
        value = final_histological_labels[int(directory)]
        if value != 'none':
            for image in os.listdir(data_location+'/'+directory):
                if value == 'lobular':
                    lobular_images.append(data_location+'/'+directory+'/'+image)
                elif value == 'ductal':
                    ductal_images.append(data_location+'/'+directory+'/'+image)

print("Finished processing {} lobular images and {} ductal images".format(len(lobular_images), len(ductal_images)))

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return np.array(a)[p.astype(int)], np.array(b)[p.astype(int)]

ext_test_x = lobular_images+ductal_images
ext_test_y = np.append(np.zeros(len(lobular_images)), np.ones(len(ductal_images)))

ext_test_x, ext_test_y = unison_shuffled_copies(ext_test_x, ext_test_y)

ext_test_x_balanced = []
ext_test_y_balanced = []

for i in ductal_images:
    ext_test_x_balanced.append(i)
    ext_test_y_balanced.append(1)

for num in range(int(len(ductal_images)/len(lobular_images))):
    for i in lobular_images:
        ext_test_x_balanced.append(i)
        ext_test_y_balanced.append(0)

ext_test_x_balanced, ext_test_y_balanced = unison_shuffled_copies(ext_test_x_balanced, ext_test_y_balanced)

train_x_modified, train_y_modified = unison_shuffled_copies(train_x_modified, train_y_modified)

#Once all metadata was extracted now we load the images into the HDF5 File
hdf5_path = './subtypeHDF5.hdf5'

trainInputFilename="./train.txt"
valInputFilename="./val.txt"
testInputFilename="./test.txt"

imagePath="./images/"

trainImageFilenames = train_x
trainImageClasses = train_y

trainImageFilenames_modified = train_x_modified
trainImageClasses_modified = train_y_modified

valImageFilenames = val_x
valImageClasses = val_y
testImageFilenames = test_x 
testImageClasses = test_y
exttestImageFilenames = ext_test_x
exttestImageClasses = ext_test_y


img_dtype = tables.UInt8Atom()
data_shape = (0, 224, 224, 3)

# open the specified hdf5 file and create arrays to store the images
hdf5_file = tables.open_file(hdf5_path, mode='w')
train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
train_storage_modified = hdf5_file.create_earray(hdf5_file.root, 'train_img_modified', img_dtype, shape=data_shape)
train_storage_pos = hdf5_file.create_earray(hdf5_file.root, 'train_img_pos', img_dtype, shape=data_shape)
train_storage_neg = hdf5_file.create_earray(hdf5_file.root, 'train_img_neg', img_dtype, shape=data_shape)
val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)
test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)
ext_test_storage = hdf5_file.create_earray(hdf5_file.root, 'ext_test_img', img_dtype, shape=data_shape)
mean_storage = hdf5_file.create_earray(hdf5_file.root, 'subtype_train_mean', img_dtype, shape=data_shape)
ext_test_storage_balanced = hdf5_file.create_earray(hdf5_file.root, 'ext_test_img_balanced', img_dtype, shape=data_shape)

# create the arrays for the outcome labels
hdf5_file.create_array(hdf5_file.root, 'train_labels', trainImageClasses)
hdf5_file.create_array(hdf5_file.root, 'train_labels_modified', trainImageClasses_modified)
hdf5_file.create_array(hdf5_file.root, 'val_labels', valImageClasses)
hdf5_file.create_array(hdf5_file.root, 'test_labels', testImageClasses)
hdf5_file.create_array(hdf5_file.root, 'exttest_labels', exttestImageClasses)
hdf5_file.create_array(hdf5_file.root, 'exttest_labels_balanced',ext_test_y_balanced) 

#Now load and store the images
def load_images(filenames, to_store,name=None):
    for i in tqdm(range(len(filenames)), total = len(filenames)):
        if i % 1000 == 0 and i > 1:
            print('Processed training data: {}/{}'.format(i, len(filenames)))
        addr = filenames[i]
        img = cv2.imread(addr)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        to_store.append(img[None])


load_images(trainImageFilenames, train_storage, "Training Data")
load_images(trainImageFilenames_modified, train_storage_modified)
load_images(train_x_pos, train_storage_pos)
load_images(train_x_neg, train_storage_neg)
load_images(valImageFilenames, val_storage)
load_images(testImageFilenames, test_storage)
load_images(exttestImageFilenames, ext_test_storage)
load_images(ext_test_x_balanced, ext_test_storage_balanced)

# close the hdf5 file
hdf5_file.close()



