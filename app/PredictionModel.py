# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:15:25 2023

@author: CSilvestre
"""
from datetime import datetime
import uuid

class PredictionModel:
    
    def __init__(self):
        self.uuid = str(uuid.uuid4())
        pass
    
    def Setup(self, name, description, model, variables, MSE, R2):
        self.name = name
        self.description = description
        self.model = model
        self.datetime = datetime.now()
        self.variables = variables
        self.MSE = MSE
        self.R2 = R2
    
    def SetModelPath(self, path):
        self.modelPath = path
        
class InputFeature:
    
    def __init__(self, name, varType, description):
        self.name = name
        self.varType = varType
        self.description = description
    
    def setName(self, name):
        self.name = name
    
    def setVarType(self, varType):
        self.varType = varType
    
    def setDescription(self, description):
        self.description = description
    
class ModelInformation:

    def __init__(self, uuid, name, description):
        self.uuid = uuid
        self.name = name
        self.description = description