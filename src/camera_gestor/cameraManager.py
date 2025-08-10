import cv2

class Camera:
    def __init__(self):#constructor
        self.WIN_SIZE=(800,600)

        self.cap=cv2.VideoCapture(0)#instaciamos el programa para que empiece a grabar
        if not self.cap.isOpened():
            raise Exception("Error no se pudo abrir la camara, verifica")
        
        print("Iniciando la camara...")
    
    
    def getFrame(self):#devolver frame de la camara
        print("obteniendo frame...")
        ret,frame=self.cap.read()
        if not ret:
            raise Exception("No se pudo obtener frame...")
        
        frame=cv2.resize(frame,self.WIN_SIZE)
        return frame
    
    def showCamera(self,image):
        cv2.imshow("Vision",image)

    def release(self):#libera la camara cuando sea necesario
        self.cap.release()
        cv2.destroyAllWindows()
        print("camara cerrada")

    def putMessage(self,image,letra):
        cv2.putText(image,f"Letra {letra}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2)


    def quitCamera(self):#cerramos la ventana
        return cv2.waitKey(5) & 0xFF == ord('q')
