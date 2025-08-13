import pandas as pd
import os

class DataGestor:
    def __init__(self):
        self.letra="A"
        self.keypoints=[]
        self.columns=self.defineColumns()
        self.data=[]

    def defineColumns(self):
        return [f"{axis}{i}"for i in range(21) for axis in['x','y','z']]+["label"]
    
    def addDatapoint(self,keypoints):
        if len(keypoints)!=63:
            print("se esperaba 63 valores")
        
        self.keypoints=keypoints
        self.data.append(self.keypoints+[self.letra])
        self.saveInCSV()
    
    def saveInCSV(self):
        filename="keypoints.csv"
        # Verificar si el archivo existe para determinar si necesitamos 
        file_exists = os.path.exists(filename) and os.stat(filename).st_size > 0
        df=pd.DataFrame(self.data,columns=self.columns)
        df.to_csv(filename,mode="a",header=not file_exists, index=False)