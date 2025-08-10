import tensorflow as tf
import os
import numpy as np
class ModelHands:
    def __init__(self):
        self.dsPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/dataset"))
        self.class_names=[]

    def getDataSet(self): #con esta funcion recopilamos los datos del dataset y los almacenamos en la variable train_ds
        train_ds=tf.keras.utils.image_dataset_from_directory(
            self.dsPath,
            image_size=(128,128),
            batch_size=32
        )
        return train_ds
    
    def trainModel(self):
        #normalizamos las imagenes
        get_train_ds=self.getDataSet()
        self.class_names=get_train_ds.class_names

        #guardamos el indice de las letras
        np.save("class_names.npy",self.class_names)
        
        train_ds=get_train_ds.map(lambda x,y:(x/255.0,y)) #mapeamos las carpetas donde estan almacenadas las imagenes y dichas imagenes las dividivos po 255 para qeu nos de o 0 o 1
        #modelo de la red neuronal
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])

        #parametros de entrenamiento del modelo
        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        
        #le decimos al modelo que revise el dataset 10 veces 
        model.fit(train_ds, epochs=10)

        # Guardar el modelo entrenado
        model.save("hand_model.h5")

if __name__=="__main__":
    modelo=ModelHands()
    modelo.trainModel()