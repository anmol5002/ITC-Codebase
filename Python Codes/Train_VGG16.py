import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy                                # Various libraries imported from tensorflow and keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train_path = 'C:/Users/itc.DESKTOP-RF2G1DL/Desktop/cam2_training_lid/train/'            # training path needs to be specified where both bad and good images are present in folders
valid_path = 'C:/Users/itc.DESKTOP-RF2G1DL/Desktop/cam2_training_lid/validation/'       # similarly validation images and test image path needs to be specified
test_path = 'C:/Users/itc.DESKTOP-RF2G1DL/Desktop/cam2_training_lid/test/'

Batch_sizes=32
save_after=20              # save model after this much epochs

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['bad', 'good'], batch_size=Batch_sizes)    # Photos are fed as a batch in training i.e. here 32 images are processed per batch
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['bad', 'good'], batch_size=Batch_sizes)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['bad', 'good'], batch_size=Batch_sizes,shuffle=False) # If (shuffle = False) then images are not shuffled 

vgg16_model = keras.applications.vgg16.VGG16()               # VGG Model is imported which is available at default path i.e. ‘C:\Users\itc.DESKTOP-RF2G1DL\______’ 
                                                             # If it is not available at default location this will automatically download (Time-consuming)
vgg16_model.layers.pop()                                     # last layer of model is scrapped

train_labels=train_batches.classes                           # Print number to be printed after training (Each class is given a number) Eg. (Bad:'0', Good:'1')

model = Sequential()                                         # Model is defined as a sequential layer for development 
for layer in vgg16_model.layers:
    model.add(layer)

for layer in model.layers:                                   # All layers in initial model are not trained and hence layer.trainable = False
    layer.trainable = False

model.add(Dense(2, activation='softmax'))                    # Add our own layer where only 2 outputs are there (Train only this layer)

model.compile(Adam(lr=.05), loss='categorical_crossentropy', metrics=['accuracy'])                           #
                                                                                                             #  Hyperparameters 
mc = keras.callbacks.ModelCheckpoint('cam2_lid{epoch:06d}.h5', save_weights_only=True, period=save_after)    #  More description in Notes
                                                                                                             #
model.fit_generator(train_batches, steps_per_epoch=157,validation_data=valid_batches, validation_steps=6, epochs=150, verbose=1,callbacks=[mc]) # Training starts


########################## Training is complete and model saved ######################################################
################### Comment this section if model already validated with a good enough dataset #######################
############################ More info in appendix ###################################################################

# predictions=model.predict_generator(test_batches,steps=7,verbose=1)                    # Predict results based on model developed through training , use no. of steps = total no of images in test set / Batch size

# cm=confusion_matrix(test_labels,predictions.argmax(axis=1)) 
 
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)                                                                       # Make a plot of confusion matrix
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# cm_plot_labels=['0','1']                                                                   # 0 represents bad and 1 represents good
# plot_confusion_matrix(cm,cm_plot_labels)                                                   # call the above function for making confusion matrix 
