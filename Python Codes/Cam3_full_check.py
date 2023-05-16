import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy                        # initial import statements
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

vgg16_model = keras.applications.vgg16.VGG16()                             # VGG Model initialised

vgg16_model.layers.pop()

model_short = Sequential()
for layer in vgg16_model.layers:
    model_short.add(layer)

model_long = Sequential()
for layer in vgg16_model.layers:
    model_long.add(layer)

model_label = Sequential()
for layer in vgg16_model.layers:
    model_label.add(layer)

for layer in model_short.layers:
    layer.trainable = False

for layer in model_long.layers:                                # As training is done previously all layers are freezed
    layer.trainable = False

for layer in model_label.layers:
    layer.trainable = False

model_short.add(Dense(2, activation='softmax'))                # Initialise models for different areas
model_long.add(Dense(2, activation='softmax'))
model_label.add(Dense(3, activation='softmax'))

model_short.load_weights("weights000060.h5")                  # Short Mitre model                 # load saved models of different areas
model_long.load_weights("weights_long000060.h5")              # Long Mitre model
model_label.load_weights("weights_label1000000030.h5")        # Label model

import os
import glob2
import time
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model                                   # import statments for camera setup and online running
import elementpath
import xml.etree.ElementTree as ET
import os
import re
import cv2 as cv
from harvesters.core import Harvester
import atexit
import cv2


camera_id = '700004978993'
iterator = 0
save_file_path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\full_all'                         # All photos are saved at this address
if not os.path.exists(save_file_path):
    os.mkdir(save_file_path)
def close_camera(ia):
    if ia != None:
        ia.stop_image_acquisition()
try:
    ia = None
    atexit.register(close_camera, ia=ia)

    
    def image_saver(img):
        global iterator
        image_path = os.path.join(save_file_path, str(iterator) + '.jpg')                # function to save main image                 
        cv.imwrite(image_path, img)
        iterator += 1
    img_dim_for_features=224                                                             # All images need to be saved as 224*224 pixels before processing
    def box(img,array):
        pts1 = np.float32(array)
        pts2 = np.float32([[0,0],[img_dim_for_features,0],[img_dim_for_features,img_dim_for_features],[0,img_dim_for_features]])

        M = cv.getPerspectiveTransform(pts1,pts2)
        M= cv.warpPerspective(img,M,(img_dim_for_features,img_dim_for_features))
        return M
    def initCamera(set_dimensions, ia):        
        ia.device.node_map.PixelFormat.value = 'BayerRG8'
        ia.device.node_map.Gain.value =45
        ia.device.node_map.ExposureTime.value = 700
        ia.device.node_map.TriggerMode.value = 'On'
        ia.device.node_map.TriggerDelay.value = 100000                                   # Camera settings
        if set_dimensions:
            ia.device.node_map.Width = 992
            ia.device.node_map.Height = 600           
            ia.device.node_map.OffsetX = 528
            ia.device.node_map.OffsetY = 382
        ia.start_image_acquisition()
        return ia
    print("working")
    h = Harvester()
    h.add_cti_file(r"bgapi2_gige.cti")
    h.update_device_info_list()
    selected_cam = -1
    for cam in h.device_info_list:
        if cam.serial_number == camera_id:
            selected_cam = h.device_info_list.index(cam)
    if selected_cam > -1:
        ia = h.create_image_acquirer(selected_cam)
    else:
        input('Camera not connected. Press Enter to exit.')
        exit()

    try:
        ia = initCamera(True, ia)
    except:
        if ia is not None:
            ia.stop_image_acquisition()
        ia = initCamera(False, ia)
    image = None


    def return_img_numpy_array():
        global ia
        global time_start
        global image
        with ia.fetch_buffer() as buffer:
            if len(buffer.payload.components) > 0:
                component = buffer.payload.components[0]

                image = component.data.reshape(
                    component.height, component.width
                )
                image = cv.cvtColor(image, cv.COLOR_BAYER_RG2RGB)
                dim = (992, 600)
                resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
                return resized
            return None


    while True:
        num=0
        for i in range(1):
            image = return_img_numpy_array()                             # image = full_size image
            image_saver(image)
            arr_short=[[740,21],[956,21],[956,256],[740,256]]
            dst=box(image,arr_short)                                     # dst = short mitre image
            arr_long=[[440,320],[822,78],[869,292],[484,585]]
            dst1=box(image,arr_long)                                     # dst1 = long mitre image          # areas are clipped from main image
            arr_label=[[0,260],[525,260],[525,590],[0,590]]
            dst2=box(image,arr_label)                                    # dst2 = label image
            path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\short\good\a'
            cv.imwrite(path+'pic'+str(num)+'.jpg', dst)
            path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\long\good\a'                  # Clipped areas saved to respective locations
            cv.imwrite(path+'pic'+str(num)+'.jpg', dst1)
            path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\label\good\a'
            cv.imwrite(path+'pic'+str(num)+'.jpg', dst2)
            num+=1
        valid_path_short='C:/Users/itc.DESKTOP-RF2G1DL/Desktop/data/short'
        valid_path_long='C:/Users/itc.DESKTOP-RF2G1DL/Desktop/data/long'
        valid_path_label='C:/Users/itc.DESKTOP-RF2G1DL/Desktop/data/label'

        image_data_short =ImageDataGenerator().flow_from_directory(valid_path_short, target_size=(224,224), classes=['bad', 'good'], batch_size=1)
        image_data_long =ImageDataGenerator().flow_from_directory(valid_path_long, target_size=(224,224), classes=['bad', 'good'], batch_size=1)
        image_data_label =ImageDataGenerator().flow_from_directory(valid_path_label, target_size=(224,224), classes=['bad', 'good'], batch_size=1)

        print("directory_made")

        predictions=model_short.predict_generator(image_data_short,steps=1,verbose=0)
        predictions_short=predictions.argmax(axis=1)                                       # predictions_short contains result of short mitre defected or non_defective
        predictions=model_long.predict_generator(image_data_long,steps=1,verbose=0)
        predictions_long=predictions.argmax(axis=1)                                        # predictions_long contains result of long mitre defected or non_defective
        predictions=model_label.predict_generator(image_data_label,steps=1,verbose=0)
        predictions_label=predictions.argmax(axis=1)                                       # predictions_label contains result of label defected or non_defective

        # if predictions_short : '0':: 'defective' ; '1':: 'non_defective'
        # if predictions_long : '0':: 'defective' ; '1':: 'non_defective'
        # if predictions_label : '0':: 'blank' ; '1':: 'defective' ; '2':: 'non_defective'


        numm=0
        img_dir = r"C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\short\good" # Save the images of each area in different directory
        data_path = os.path.join(img_dir,'*g')
        files = glob2.glob(data_path)
        data = []
        for f1 in files:
            if predictions_short[numm] == 1:
                path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\other\n_short\a'
                dst= cv.imread(f1)
                cv.imwrite(path+'pic'+str(iterator)+'.jpg', dst)
                
            else:
                path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\other\d_short\a'
                dst= cv.imread(f1)
                cv.imwrite(path+'pic'+str(iterator)+'.jpg', dst)   
            numm=numm+1
            
        numm=0
        img_dir = r"C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\long\good" # Save the images of each area in different directory
        data_path = os.path.join(img_dir,'*g')
        files = glob2.glob(data_path)
        data = []
        for f1 in files:
            if predictions_long[numm] == 1:
                path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\other\n_long\a'
                dst= cv.imread(f1)
                cv.imwrite(path+'pic'+str(iterator)+'.jpg', dst1)
                
            else:
                path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\other\d_long\a'
                dst= cv.imread(f1)
                cv.imwrite(path+'pic'+str(iterator)+'.jpg', dst1)   
            numm=numm+1
            
        numm=0
        img_dir = r"C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\label\good" # Save the images of each area in different directory 
        data_path = os.path.join(img_dir,'*g')
        files = glob2.glob(data_path)
        for f1 in files:
            if predictions_label[numm] == 2:
                path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\other\n_label\a'
                dst= cv.imread(f1)
                cv.imwrite(path+'pic'+str(iterator)+'.jpg', dst2)
                
            elif predictions_label[numm] == 1:
                path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\other\d_label\a'
                dst= cv.imread(f1)
                cv.imwrite(path+'pic'+str(iterator)+'.jpg', dst2)
                
            else:
                path = r'C:\Users\itc.DESKTOP-RF2G1DL\Desktop\data\other\b_label\a'
                dst= cv.imread(f1)
                cv.imwrite(path+'pic'+str(iterator)+'.jpg', dst2)   
            numm=numm+1


    ia.stop_image_acquisition()

except Exception as e:
    if ia is not None:
        ia.stop_image_acquisition()
    print("Exception", str(e))
    input('press enter to exit')

exit(0)

