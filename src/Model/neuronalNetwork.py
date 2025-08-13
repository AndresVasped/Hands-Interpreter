from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

class ModelHands:
    def __init__(self):
        #baseDir=os.path.dirname(__file__)
        #self.dsPath = os.path.normpath(os.path.join(baseDir, "../../data/dataset"))
        #self.dsPath=f"C:\Users\JAVS\Desktop\Trabajos de universidad\sexto semestre\arquitectura\Hands Interpreter\data\dataset\A"
        #print(self.dsPath)
        pass


    def getDataSet(self,coordinates,labels): #con esta funcion recopilamos los datos del dataset
        #convertimos las listas a arreglos numpy
        X=np.array(coordinates,dtype=np.float32)
        y=np.array(labels,dtype=np.int32)
        print(f"   X shape: {X.shape}")  # (n_samples, 63)
        print(f"   y shape: {y.shape}")

        #normalizar las coordenadas
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        #convertimos los labels a categorical
        num_classes = len(np.unique(y))
        y_categorical = keras.utils.to_categorical(y, num_classes)

        #entrenbamiento y prueba los dividimos
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_categorical, 
            test_size=0.2,          # 20% para pruebas
            random_state=42,        # Para reproducibilidad
            stratify=y              # Mantener proporción de clases
        )
        return X_train, X_test, y_train, y_test,num_classes,scaler

        
    def trainModel(self,coordinates,labels):
        
        X_train, X_test, y_train, y_test,num_classes,scaler=self.getDataSet(coordinates,labels)

        #modelo de la red neuronal
        model= keras.Sequential([
            # Capa de entrada
            layers.Input(shape=(63,)),

            # Capas ocultas
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),

            #capa de salida
            layers.Dense(num_classes, activation='softmax')
        ])

        
        #parametros de entrenamiento del modelo
        model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        
        model.summary()
        #le decimos al modelo que revise el dataset 10 veces 
        history=model.fit(X_train,y_train,epochs=100, batch_size=32,validation_data=(X_test, y_test),verbose=1,callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=15,           # Parar si no mejora en 15 épocas
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5,            # Reducir learning rate
                    patience=10
                )
            ]
        )
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Precisión en prueba: {test_accuracy:.2%}")

        # Guardar el modelo entrenado
        model.save("hand_model.keras")

        #guardamos el scaler
        with open("scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        plot_training_history(history)

def plot_training_history(self,history):
    """
    Grafica la historia del entrenamiento
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Precisión
    ax1.plot(history.history['accuracy'], label='Entrenamiento')
    ax1.plot(history.history['val_accuracy'], label='Validación')
    ax1.set_title('Precisión del Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Precisión')
    ax1.legend()
    ax1.grid(True)
    
    # Pérdida
    ax2.plot(history.history['loss'], label='Entrenamiento')
    ax2.plot(history.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida del Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    modelo=ModelHands()
    history=modelo.trainModel()
