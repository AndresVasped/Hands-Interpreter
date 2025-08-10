import mediapipe as mp
import numpy as np
import tensorflow as tf
import cv2
import os
from src.camera_gestor.saveDataset import SaveInDs
class MediapipeManager():
    
    def __init__(self):
        self.mp_hands=mp.solutions.hands #es un modulo ya entrenado para el seguimiento de manos
        self.mp_drawing=mp.solutions.drawing_utils #dibujamos las 21 lanmraks de la mano

        self.manos=self.mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5)#iniciamos el modelo de las manos
        self.model=tf.keras.models.load_model(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../hand_model.h5")))
        self.class_names=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../class_names.npy"))

    def processFrame(self,frame):
        #convertimos a rgb
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesar la imagen para detectar las manos
        results = self.manos.process(image_rgb)
        return results
    
    def drawHands(self,frame,results):
        letra=None
        #Dibuja las landmarks de las manos detectadas
        image = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                #vamos a delimitar la mano para procesarla por el modelo
                h,w,_=frame.shape

                coordenadasX= [lm.x * w for lm in hand_landmarks.landmark]#obtenemos las coordenadas de la mano
                coordenadasY= [lm.y * h for lm in hand_landmarks.landmark]

                #obtenemos el maximo y minimo de estas coordenadas
                xmin, xmax = int(min(coordenadasX)), int(max(coordenadasX))
                ymin, ymax = int(min(coordenadasY)), int(max(coordenadasY))

                cv2.rectangle(frame, (xmin-40, ymin-40), (xmax+40, ymax+40), (0, 255, 0), 2)
                
                #recortamos la imagen
                hand_img = frame[ymin-40:ymax+40, xmin-40:xmax+40]
                cv2.imshow("Recorte",hand_img)

                #sistema de recoleccion de datos
                #svds=SaveInDs()
                #svds.saveInds(hand_img)
                
                if hand_img.size==0:#si la pocision es 0 continua y no craseha el programa
                    continue
                hand_img = cv2.resize(hand_img, (128, 128))
                hand_img = hand_img / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                # verificamos con el modelo
                prediccion=self.model.predict(hand_img)
                clase_index = np.argmax(prediccion)
                letra=self.class_names[clase_index]

                # Dibujar todas las landmarks
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return image,letra
    
    def release(self):#liberamos al modelo
        self.manos.close()
