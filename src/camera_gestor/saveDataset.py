import cv2
import os
import re
class SaveInDs:
    def __init__(self):
        self.path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/dataset/C"))
        os.makedirs(self.path, exist_ok=True) #si no existe
        self.contador=self.getLastNumber()+1

    def getLastNumber(self):
        archivos = os.listdir(self.path)
        numeros = [int(re.match(r'c_(\d+)\.', f, re.IGNORECASE).group(1)) 
            for f in archivos if re.match(r'c_(\d+)\.', f, re.IGNORECASE)]
        return max(numeros) if numeros else 0

    def saveInds(self,frame):
        file_path=os.path.join(self.path, f"C_{self.contador}.jpg")
        cv2.imwrite(file_path,frame)

        print(self.contador)