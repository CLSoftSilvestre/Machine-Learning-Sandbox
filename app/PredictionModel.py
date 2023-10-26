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
        self.modelType = "regression"
        self.pVariable = "not defined"
        self.pVariableUnits = "not defined"
    
    def SetModelPath(self, path):
        self.modelPath = path
    
    def SetTrainImage(self, imageData):
        self.imageData = imageData
    
    def SetCorrelationMatrixImage(self, imageData):
        self.correlationMatrixImage = imageData
    
    def SetModelVersion(self, modelVersion, appVersion):
        self.modelVersion = modelVersion
        self.appversion = appVersion
    
    def SetModelType(self, modelType):
        self.modelType = modelType
    
    def SetPredictVariable(self, pVariable, pVariableUnits):
        self.pVariable = pVariable
        self.pVariableUnits = pVariableUnits
        
class InputFeature:
    
    def __init__(self, name, varType, description):
        self.name = name
        self.varType = varType
        self.description = description
        self.importance = 0
        self.unit = "Not defined"
    
    def setName(self, name):
        self.name = name
    
    def setVarType(self, varType):
        self.varType = varType
    
    def setDescription(self, description):
        self.description = description

    def setValue(self, value):
        self.value = value
    
    def setImportance(self, value):
        self.importance = value
    
    def setDescribe(self, value):
        self.describe = value
    
    def setUnit(self, value):
        self.unit = value
    
    def serialize(self):
        return {
            'name': self.name, 
            'type': self.varType,
            'unit': self.unit,
            'min': self.describe['min'],
            'max': self.describe['max'] 
        }
    
class ModelInformation:
    
    def __init__(self, uuid, name, description):
        self.uuid = uuid
        self.name = name
        self.description = description

class ReturnFeature:
    
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
    def serialize(self):
        return {
            'name': self.name, 
            'value': self.value
        }

class Prediction:
    def __init__(self, value, features):
        self.value = value
        self.features = features

class PredictionBatch:
    def __init__(self, model, predictions):
        self.model = model
        self.prediction = predictions