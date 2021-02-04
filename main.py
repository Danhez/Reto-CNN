import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, VGG19
from tensorflow.keras.utils import to_categorical

import numpy as np



#Uso de VRAM
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
l_relu = LeakyReLU()

def Entrenar():

    ruta_dataset_entrenamiento = "Dataset/train"


    num_classes = 6

    # Preprocesamiento
    #Data aumentation
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


    #Load Data
    train_generator = train_datagen.flow_from_directory(ruta_dataset_entrenamiento, target_size=(150,150), color_mode='rgb', batch_size=32, class_mode='categorical', shuffle=True)


    print(".........Dataset Cargado..........")

    #Design
    # Se define como un modelo secuencial
    model = Sequential()

    print("..........Construyendo Modelo.........")

    # Se añaden las capas y sus hiperparámetros
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(150, 150, 3), activation=l_relu))
    model.add(MaxPooling2D(pool_size=(2, 2))) #75x75
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3,3), padding='same', activation=l_relu))
    model.add(MaxPooling2D(pool_size=(2, 2))) #37x37
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (2, 2), padding='same', activation=l_relu))
    model.add(MaxPooling2D(pool_size=(2, 2))) #18x18
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (2, 2), padding='same', activation=l_relu))
    model.add(MaxPooling2D(pool_size=(2, 2))) #9x9
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    # La capa de salida debe tener el mismo número de clases
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()


    #Compilation
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    print("..........Modelo Compilado.............")

    #Training
    # Funciones callbacks
    callbacks = [EarlyStopping(monitor='loss', mode='min', verbose=1,patience=4), ModelCheckpoint(filepath="Dataset/Checkpoint/SightA.h5", monitor='loss', save_best_only=True, verbose=1,mode='min')]


    step_size_train=train_generator.n/train_generator.batch_size
    history = model.fit(x=train_generator, steps_per_epoch=step_size_train, epochs=42, callbacks=callbacks)


def Clasificar():
    ruta_dataset_prueba = "Dataset/test/images"
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(ruta_dataset_prueba,target_size=(150,150), color_mode='rgb', batch_size=1, class_mode='categorical', shuffle=False)

    model_loaded = load_model("Dataset/Checkpoint/SightA.h5", custom_objects={'LeakyReLU': LeakyReLU()})

    step_size_test = test_generator.n / test_generator.batch_size

    y_pred_prob = model_loaded.predict(x=test_generator, steps=step_size_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    nombres = test_generator.filenames


    resultados = {}
    for _i in range(len(nombres)):
        if y_pred_classes[_i] == 0:
            resultados[nombres[_i]] = "Bosque"
        elif y_pred_classes[_i] == 1:
            resultados[nombres[_i]] = "Calle"
        elif y_pred_classes[_i] == 2:
            resultados[nombres[_i]] = "Edificio"
        elif y_pred_classes[_i] == 3:
            resultados[nombres[_i]] = "Glaciar"
        elif y_pred_classes[_i] == 4:
            resultados[nombres[_i]] = "Montaña"
        elif y_pred_classes[_i] == 5:
            resultados[nombres[_i]] = "Oceano"
    print("******************SightA******************")
    print(y_pred_classes)
    print(resultados)

    print("\n******************MobileNetV2******************")
    mobile = MobileNetV2(input_shape=None, alpha=1.0, include_top=False, weights='imagenet',input_tensor=None, pooling=None, classes=6,classifier_activation='softmax')
    mobile.compile()
    mobile_prob = mobile.predict(x=test_generator, steps=step_size_test)
    mobile_pred_classes = np.argmax(mobile_prob, axis=1)
    print(mobile_pred_classes)

    print("\n******************ResNetV50V2******************")
    res = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None,input_shape=None, pooling=None, classes=6,classifier_activation='softmax')
    res.compile()
    res_prob = res.predict(x=test_generator, steps=step_size_test)
    res_pred_classes = np.argmax(res_prob, axis=1)
    print(res_pred_classes)

    print("\n******************VGG19******************")
    vgg = VGG19(include_top=False, weights='imagenet', input_tensor=None,input_shape=None, pooling=None, classes=6,classifier_activation='softmax')
    vgg.compile()
    vgg_prob = vgg.predict(x=test_generator, steps=step_size_test)
    vgg_pred_classes = np.argmax(vgg_prob, axis=1)
    print(vgg_pred_classes)


if __name__ == '__main__':
    #Entrenar()
    Clasificar()
