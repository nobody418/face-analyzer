#OpenCV module
import cv2
#Modulo para leer directorios y rutas de archivos
import os
from tqdm import tqdm
import dlib
import numpy as np
import pandas as pd
import random
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50, InceptionV3, InceptionResNetV2, VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import GlobalMaxPooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import gradient_descent_v2 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras import optimizers 
import argparse  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


train = "Base de datos/"
#test = ruta + "/BD/Test"
batch_size=64
img_size=150 # 150 x 150
img_width, img_height = img_size, img_size
training_data = []
test_data = []
names_array_train= []
names_array_test= []

ruta2="Modelos/"
Haar = ruta2+"haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(Haar)
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(ruta2 + "shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1(ruta2 + "/dlib_face_recognition_resnet_model_v1.dat")
def numero_clases(ruta):
    clases = 0
    clases = len(os.listdir(ruta))
    return clases

def create_dataset(ruta, array, array_n):
  size = 1
  cont_aux=0
  for category in os.listdir(ruta):  # do dogs and cats
      cont=0
      cont_aux=cont_aux+1
      path = os.path.join(ruta, category)  # create path to dogs and cats
      class_num = os.listdir(ruta).index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
      array_n.append([category,class_num])
      for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
          #img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
          #img_array = cv2.imread(os.path.join(path,img))  # convert to array
          img_array = dlib.load_rgb_image(os.path.join(path,img))
          img_detected = detector(img_array, 1)
          if len(img_detected) < 2 and len(img_detected) > 0:
            cont=cont+1
            img_shape = sp(img_array,img_detected[0])
            face_resize  = dlib.get_face_chip(img_array,img_shape)
            img_rep = facerec.compute_face_descriptor(face_resize)
            face_resize = dlib.as_grayscale(face_resize)
            img_representation = np.array(img_rep)
            face_resize = face_resize.flatten()
            new_array=np.array(face_resize)
            array.append([new_array, img_representation, class_num])  # guardar en array el vector de la imagen y el npumero de la clase
      print(str(cont_aux)+". "+category)
      print('numero de fotos:' + str(cont))
      print('numero de fotos total:' + str(len(array)))
      print(" ")

training_data = []
test_data = []
names_array_train= []
names_array_test= []
cant_clases = numero_clases(train)
create_dataset(train, training_data, names_array_train)

name_class = []
label_class = []

for features,label in names_array_train:
    name_class.append(features)
    label_class.append(label)

df = pd.DataFrame(name_class)
df.to_csv(ruta2 + '/InceptionResNET3P(prueba_55personas).csv', sep=',',header=False)

random.shuffle(test_data)
random.shuffle(training_data)

X_train_1 = []
X_train = []
Y_train = []

for img,features,label in training_data:
    X_train.append(img)
    X_train_1.append(features)
    Y_train.append(label)

X_train = np.array(X_train)
X_train = np.array(X_train).reshape(-1, img_size, img_size, 1)
X_train = np.repeat(X_train, 3, axis=-1)
X_train_1 = np.array(X_train_1)

X_train = X_train/255.0

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = cant_clases)

train_images, val_images, train_features, val_features, train_labels, val_labels = train_test_split(X_train, X_train_1 , Y_train, test_size=0.30)

input = Input(shape=(img_size,img_size,3))
model_original = InceptionResNetV2(include_top = False, input_shape=(img_height,img_width,3), weights = 'imagenet')
for layer in model_original.layers:
    layer.trainable = False

print("All trainable layers in false...")

input2 = tf.keras.layers.Input(shape=(128,))
flat_layer = layers.GlobalAveragePooling2D()(model_original.output)
flat_layer = layers.Flatten()(flat_layer)
flat_layer2 = layers.concatenate([flat_layer, input2])
dense_layer = layers.BatchNormalization()(flat_layer2)
dense_layer_1 = layers.Dense(2048, activation='relu')(dense_layer)
dense_layer_1 = layers.Dropout(0.5)(dense_layer_1)
dense_layer_1 = layers.BatchNormalization()(dense_layer_1)
dense_layer_1 = layers.Dense(1024, activation='relu')(dense_layer_1)
dense_layer_1 = layers.Dropout(0.2)(dense_layer_1)
dense_layer_1 = layers.BatchNormalization()(dense_layer_1)
dense_layer_1 = layers.BatchNormalization()(flat_layer2)
dense_layer_3 = layers.Dense(cant_clases, activation='softmax')(dense_layer_1)
model = Model(inputs = [model_original.inputs, input2], outputs = dense_layer_3)
print("\nNumber of layers en ResNet= ",len(model_original.layers))

model.compile(
         loss  = tf.keras.losses.CategoricalCrossentropy(),
          metrics = tf.keras.metrics.CategoricalAccuracy(),
          #optimizer = gradient_descent_v2.SGD(0.0001))
          optimizer = Adam(0.0001))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=3)

datagen = ImageDataGenerator(zoom_range = 0.2, shear_range=0.2)
datagen.fit(train_images)

t1 = time.time()
history = model.fit([train_images, train_features], train_labels, epochs=30, validation_data=([val_images, val_features], val_labels), callbacks= early_stopping, verbose=1)


t2 = time.time()

print("\nProcessing time %0.3f minutes" % ((t2-t1)/60))
model.save(ruta2 + '/modelo_entrenado_InceptionResNET3P(prueba2).h5')
