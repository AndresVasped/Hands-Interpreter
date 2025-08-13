import mediapipe as mp
import numpy as np
import tensorflow as tf
import cv2
import os
from src.camera_gestor.saveDataset import SaveInDs
from src.Model.neuronalNetwork import ModelHands
from sklearn.preprocessing import StandardScaler
import pickle

class MediapipeManager():
    
    def __init__(self):
        self.mp_hands=mp.solutions.hands #es un modulo ya entrenado para el seguimiento de manos
        self.mp_drawing=mp.solutions.drawing_utils #dibujamos las 21 lanmraks de la mano

        self.manos=self.mp_hands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.5)#iniciamos el modelo de las manos
        
        self.model=tf.keras.models.load_model(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../hand_model.keras")))

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
               
                #self.data.addDatapoint(self.keypoints)
                
                coordenadasX= [lm.x * w for lm in hand_landmarks.landmark]#obtenemos las coordenadas de la mano
                coordenadasY= [lm.y * h for lm in hand_landmarks.landmark]

                #obtenemos el maximo y minimo de estas coordenadas
                xmin, xmax = int(min(coordenadasX)), int(max(coordenadasX))
                ymin, ymax = int(min(coordenadasY)), int(max(coordenadasY))

                cv2.rectangle(frame, (xmin-40, ymin-40), (xmax+40, ymax+40), (0, 255, 0), 2)
                
                #recortamos la imagen
                hand_img = frame[ymin-40:ymax+40, xmin-40:xmax+40]
                
                result=self.getPrediction(hand_landmarks)
                if result[0] is not None:
                    letra,confidence=result
                    print(f"Predicción: {letra} ({confidence:.1%})")

                cv2.imshow("Recorte",hand_img)

                #sistema de recoleccion de datos
                #svds=SaveInDs()
                #svds.saveInds(hand_img)
                
                """""
                if hand_img.size==0:#si la pocision es 0 continua y no craseha el programa
                    continue
                
                hand_img = cv2.resize(hand_img, (640, 640))
                hand_img = hand_img / 255.0
                hand_img = np.expand_dims(hand_img, axis=0).astype('float32')

                # verificamos con el modelo
                prediccion=self.model.predict(hand_img,conf=0.55)
                clase_index = np.argmax(prediccion)
                letra=self.class_names[clase_index]

                """""
                # Dibujar todas las landmarks
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return image,letra
    
    def load_scaler(self):
        scaler_path=(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scaler.pkl")))
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    
    def getPrediction(self,hand_landmarks):
        landmarks=[]
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        if len(landmarks)==63:
            scaler=self.load_scaler()
            landmarks_scaled = scaler.transform([landmarks])
            #predecimos el modelo
            prediction=self.model.predict(landmarks_scaled,verbose=0)
            predictClass=np.argmax(prediction[0])
            confidence=np.max(prediction[0])

            letters=['A','B']
            return letters[predictClass], confidence
        
        return None,0.0
        
    def getKeypoints(self):
        print("coordenadas: ",self.keypoints)
        return self.keypoints
    
    def getCoordinatesInImages(self):
        coordinates=[]
        labels=[]
        letterToNumber={'A':0,'B':1}

        path=os.path.normpath(os.path.join(os.path.dirname(__file__), "../../data/dataset"))
        self.manos=self.mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.7)
        
        #recorremos las carpetas
        for letter in ['A','B']:
            completePath=os.path.join(path,letter)
            if not os.path.exists(completePath):#si la carpeta no existe
                print("No existe esa ubicacion ",completePath)

            #recorremos las imagenes
            for filename in os.listdir(completePath):
                if filename.lower().endswith('jpg'):
                    imagePath=os.path.join(completePath,filename)
                    try:
                        image=cv2.imread(imagePath)
                        #convertir a escalas
                        results=self.processFrame(image)

                        #si detectamos alguna
                        if results.multi_hand_landmarks:
                            #extramemos las coordenadas
                            landmarks=[]
                            for hand_landmarks in results.multi_hand_landmarks:
                                for landmark in hand_landmarks.landmark:
                                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                            if len(landmarks)==63: #si tenemos exactamente las 63 coordenadas
                                coordinates.append(landmarks)
                                labels.append(letterToNumber[letter])#añadimos el numero de la letra que este  en letterToNumber

                        else:
                            print("no se detectaron manos en las imagenes")
                    except Exception as e:
                        print(f"Error al procesar {letter}/{filename}: {str(e)}")

        self.release()
        print("Coordenadas halladas: ",coordinates)
        print("En sus pocisiones halladas: ",labels)
        model=ModelHands()
        model.trainModel(coordinates,labels)
                
    def release(self):#liberamos al modelo
        self.manos.close()
