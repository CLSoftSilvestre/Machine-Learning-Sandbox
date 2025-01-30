# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:23:59 2023

@author: CSilvestre
"""

import pickle
import glob
import PredictionModel


class ModelManager:
    
    def __init__(self):
        pass
      
    def SaveModel(self, model, path):
        pickle.dump(model, open(path, 'wb'))
        return True
    
    def LoadModel(self, path):
        loaded_model = pickle.load(open(path, 'rb'))
        loaded_model.SetModelPath(path)
        return loaded_model
    
    def DeleteModel(self, path):
        pass
    
    def GetModelsFiles(self, path):      
        res = []
        res = glob.glob(path)
        return res
    
    def GetModelsList(self, path):
        models = []
        
        files = self.GetModelsFiles(path)
        
        for file in files:
            try:
                models.append(self.LoadModel(file))
            except Exception as Err:
                print("Error loading model " + file + " : " + str(Err))
        
        return models
        
        