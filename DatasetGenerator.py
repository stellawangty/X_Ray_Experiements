import os

import cv2
import numpy as np
import pydicom
from PIL import Image

import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                # lineItems = line.split()
                lineItems = line.split('------')

                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imagePath = imagePath
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    def transform_(self, X: np.ndarray):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        X = clahe.apply(X)
        return X

    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]

        imageData = pydicom.dcmread(imagePath).pixel_array
        imageData = np.array(imageData).astype(np.uint8)
        imageData = self.transform_(imageData)
        imageData = Image.fromarray(imageData).convert('RGB')

        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
    