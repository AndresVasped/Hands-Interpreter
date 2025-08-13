from src.camera_gestor.cameraManager import Camera
from src.camera_gestor.mediaPipeManager import MediapipeManager

import cv2

class Main:
    def __init__(self):
        self.camera=Camera()#instanciamos la camara
        self.deteccionManos=MediapipeManager()#instanciamos la deteccion de manos

    def run(self):
        ##codigo para correr el programa llamando a la clase cameraManager
        print("corriendo...")
        try:
            #self.deteccionManos.getCoordinatesInImages()
            while True:
                
                frame=self.camera.getFrame()
                result=self.deteccionManos.processFrame(frame)
                image,letra=self.deteccionManos.drawHands(frame,result)
                
                #mostramos el mensaje
                self.camera.putMessage(frame,letra)

                #self.camera.showCamera(image)#mostramos la pantalla de la camara

                #es por si solo queremos mostrar la manos sin ningun dibujo
                self.camera.showCamera(frame)

                if self.camera.quitCamera():#cerramos la camara
                    break
        finally:
            self.camera.release()
            self.deteccionManos.release()
            
            