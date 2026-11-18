from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

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
        self.plot_eda(y)
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

        X_train, y_train = self.augment_data(X_train, y_train)

        #modelo secuenquial
        """""
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
        """
        #CNN
        inputs = layers.Input(shape=(63,))
        # Convertimos el vector plano de 63 en una secuencia espacial de 21 puntos con 3 ejes (x, y, z)
        x = layers.Reshape((21, 3))(inputs)

        # Bloque Convolucional 1
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.25)(x)

        # Bloque Convolucional 2 con Conexión Residual (Técnica avanzada para Nivel 5)
        shortcut = layers.Conv1D(filters=128, kernel_size=1, padding='same')(x)
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same')(x)
        x = layers.Add()([x, shortcut]) # Conexión residual (Skip connection)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling1D()(x)

        # Clasificador Denso Final
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        #parametros de entrenamiento del modelo
        model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        
        model.summary()
        #le decimos al modelo que revise el dataset 100 veces 
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

        self.plot_training_history(history)
        self.evaluate_model(model, X_test, y_test, num_classes)

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

    def plot_eda(self, y):
        """Genera gráfico de distribución de clases para el Análisis Exploratorio (Criterio 1)"""
        letters = ['A', 'B', 'C', 'D', 'H', 'L', 'O']
        unique, counts = np.unique(y, return_counts=True)
        
        plt.figure(figsize=(8, 4))
        plt.bar([letters[i] for i in unique], counts, color='teal', edgecolor='black')
        plt.title('Distribución de Muestras por Letra (EDA)')
        plt.xlabel('Letra')
        plt.ylabel('Cantidad de Imágenes / Landmarks')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def augment_data(self, X_train, y_train):
        """Aumento de datos (Data Augmentation) inyectando ruido gaussiano y escalado a los landmarks (Criterio 1)"""
        # 1. Ruido gaussiano (simula temblor o variaciones en la detección de MediaPipe)
        noise = np.random.normal(0, 0.03, X_train.shape).astype(np.float32)
        X_noisy = X_train + noise
        
        # 2. Escalado leve (simula mano más cerca o más lejos de la cámara)
        scale = np.random.uniform(0.9, 1.1, (X_train.shape[0], 1)).astype(np.float32)
        X_scaled = X_train * scale
        
        X_aug = np.vstack([X_train, X_noisy, X_scaled])
        y_aug = np.vstack([y_train, y_train, y_train])
        print(f"Datos de entrenamiento aumentados de {len(X_train)} a {len(X_aug)} muestras.")
        return X_aug, y_aug

    def evaluate_model(self, model, X_test, y_test, num_classes):
        """Evaluación exhaustiva con Matriz de Confusión y Curvas ROC (Criterio 3)"""
        letters = ['A', 'B', 'C', 'D', 'H', 'L', 'O']
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # 1. Reporte de métricas (Precision, Recall, F1-Score)
        print("\n=== REPORTE DE CLASIFICACIÓN DETALLADO ===")
        print(classification_report(y_true, y_pred, target_names=letters[:num_classes]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 2. Matriz de Confusión
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=letters[:num_classes], yticklabels=letters[:num_classes], ax=ax1)
        ax1.set_title('Matriz de Confusión')
        ax1.set_xlabel('Predicción')
        ax1.set_ylabel('Real')

        # 3. Curvas ROC Multiclase
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, label=f'Letra {letters[i]} (AUC = {roc_auc:.2f})')

        ax2.plot([0, 1], [0, 1], 'k--', lw=1)
        ax2.set_title('Curvas ROC por Clase')
        ax2.set_xlabel('Tasa de Falsos Positivos (FPR)')
        ax2.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
        ax2.legend(loc='lower right')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    modelo=ModelHands()
    history=modelo.trainModel()
