# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:15:25 2023

@author: CSilvestre
"""
from datetime import datetime
import uuid
from DataStudio import DataStudio, DataOperation
import math
from FlowsManager import Flow
import json
import sys
from FlowsManager import NodeType, ValueType, Flow, Node, InputConnector


class PredictionModel:
    
    def __init__(self):
        self.uuid = str(uuid.uuid4())
        pass
    
    def Setup(self, name, description, keywords, model, variables, MSE, R2):
        self.name = name
        self.description = description
        self.keywords = keywords
        self.model = model
        self.datetime = datetime.now()
        self.variables = variables
        self.MSE = MSE
        self.RMSE = math.sqrt(MSE)
        self.R2 = R2
        self.modelType = "regression"
        self.pVariable = "not defined"
        self.pVariableUnits = "not defined"
        self.realTestList = list()
        self.predTestList = list()
        self.dataStudio = DataStudio()
        self.automation = Flow()
    
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

    def SetTestData(self, realTest, predTest):
        self.realTestList = realTest
        self.predTestList = predTest
        # Calculate the r^2 as accuracy if the model is classification
        if self.modelType == "classification":
            accurate = 0
            total = len(realTest)
            accuracy = 0
            for index, item in enumerate(realTest):
                if item == predTest[index]:
                    accurate +=1
            
            accuracy = (accurate / total) * 100

            self.R2 = accuracy

    def SetDataStudioData(self, data):
        self.dataStudio = data

    def SetAutomationDiagram(self, diagramData):
        self.flow = Flow()
        self.flow.SetJsonLayout(json.loads(diagramData))

        data = json.loads(diagramData)
        
        # Parse the data from the drawflow JSON file and create the elements and connections
        for i in range(100):
            try:  
                elementClass = data["drawflow"]["Home"]["data"][str(i)]["class"]
                #print(data["drawflow"]["Home"]["data"][str(i)], file=sys.stderr)
                #print("--------------------------------------", file=sys.stderr)
                #print("Element class: " + str(elementClass), file=sys.stderr)
                if elementClass == "s7connector":
                    #get data from s7 connector
                    ip = data["drawflow"]["Home"]["data"][str(i)]["data"]["s7"]["ip"]
                    rack = data["drawflow"]["Home"]["data"][str(i)]["data"]["s7"]["rack"]
                    slot = data["drawflow"]["Home"]["data"][str(i)]["data"]["s7"]["slot"]
                    family = data["drawflow"]["Home"]["data"][str(i)]["data"]["s7"]["family"]

                    params = {
                        "IP": ip,
                        "RACK": rack,
                        "SLOT": slot,
                        "FAMILY": family
                    }

                    # Create the Node and add the connections
                    node = Node(i, elementClass, params)
                    self.flow.AddNode(node)

                elif elementClass == "s7variable":
                    #get data from s7 connector
                    db = data["drawflow"]["Home"]["data"][str(i)]["data"]["db"]
                    start = data["drawflow"]["Home"]["data"][str(i)]["data"]["start"]
                    size = data["drawflow"]["Home"]["data"][str(i)]["data"]["size"]
                    connector = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]

                    params = {
                        "DB": db,
                        "START": start,
                        "SIZE": size,
                    }

                    node = Node(i, elementClass, params)
                    nodeId = connector[0]["node"]
                    nodeInp = connector[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.S7CONNECTION)
                    node.SetInputConnector(con)
                    self.flow.AddNode(node)

                elif elementClass == "mqttconnector":
                    #get data from mqtt connector
                    server = data["drawflow"]["Home"]["data"][str(i)]["data"]["mqtt"]["server"]
                    port = data["drawflow"]["Home"]["data"][str(i)]["data"]["mqtt"]["port"]
                    username = data["drawflow"]["Home"]["data"][str(i)]["data"]["mqtt"]["username"]
                    password = data["drawflow"]["Home"]["data"][str(i)]["data"]["mqtt"]["password"]

                    params = {
                        "SERVER": server,
                        "PORT": port,
                        "USERNAME": username,
                        "PASSWORD": password,
                    }

                    # Create the Node and add the connections
                    node = Node(i, elementClass, params)
                    self.flow.AddNode(node)
                
                elif elementClass == "mqtttopic":
                    #get data from mqtt topic connector
                    topic = data["drawflow"]["Home"]["data"][str(i)]["data"]["topic"]
                    qos = data["drawflow"]["Home"]["data"][str(i)]["data"]["qos"]
                    connector = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]

                    params = {
                        "TOPIC": topic,
                        "QOS": qos,
                    }

                    node = Node(i, elementClass, params)
                    nodeId = connector[0]["node"]
                    nodeInp = connector[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.MQTTCONNECTION)
                    node.SetInputConnector(con)
                    self.flow.AddNode(node)

                elif elementClass == "bleconnector":
                    #get data from ble connector
                    name = data["drawflow"]["Home"]["data"][str(i)]["data"]["name"]

                    params = {
                        "NAME": name,
                    }

                    # Create the Node and add the connections
                    node = Node(i, elementClass, params)
                    self.flow.AddNode(node)
                
                elif elementClass == "blecharacteristic":
                    #get data from ble characteristic
                    uuid = data["drawflow"]["Home"]["data"][str(i)]["data"]["uuid"]
                    connector = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]

                    params = {
                        "UUID": uuid,
                    }

                    node = Node(i, elementClass, params)
                    nodeId = connector[0]["node"]
                    nodeInp = connector[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.MQTTCONNECTION)
                    node.SetInputConnector(con)
                    self.flow.AddNode(node)
                
                elif elementClass == "influxdb":
                    bucket = data["drawflow"]["Home"]["data"][str(i)]["data"]["bucket"]
                    organization = data["drawflow"]["Home"]["data"][str(i)]["data"]["organization"]
                    token = data["drawflow"]["Home"]["data"][str(i)]["data"]["token"]
                    url = data["drawflow"]["Home"]["data"][str(i)]["data"]["url"]

                    params = {
                        "BUCKET": bucket,
                        "ORGANIZATION": organization,
                        "TOKEN": token,
                        "URL": url,
                    }

                    node = Node(i, elementClass, params)
                    self.flow.AddNode(node)

                elif elementClass == "influxpoint":
                    point = data["drawflow"]["Home"]["data"][str(i)]["data"]["point"]
                    tag = data["drawflow"]["Home"]["data"][str(i)]["data"]["tag"]
                    field = data["drawflow"]["Home"]["data"][str(i)]["data"]["filed"]
                    connector = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]

                    params = {
                        "POINT": point,
                        "TAG": tag,
                        "FIELD": field,
                    }
                    
                    node = Node(i, elementClass, params)
                    nodeId = connector[0]["node"]
                    nodeInp = connector[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.MQTTCONNECTION)
                    node.SetInputConnector(con)
                    self.flow.AddNode(node)

                elif elementClass == "static":
                    staticValue = data["drawflow"]["Home"]["data"][str(i)]["data"]["staticvalue"]

                    params = {
                        "STATICVALUE": float(staticValue),
                    }

                    node = Node(i, elementClass, params)
                    self.flow.AddNode(node)

                elif elementClass == "toggle":
                    #value = data["drawflow"]["Home"]["data"][str(i)]["data"]["value"]
                    
                    params = {
                        "VALUE": 0,
                    }
                    
                    node = Node(i, elementClass, params)
                    node.outputValue = 0
                    self.flow.AddNode(node)
                
                elif elementClass == "slider":
                    node = Node(i, elementClass, params)
                    node.outputValue = 0
                    self.flow.AddNode(node)  

                elif elementClass == "random":
                    minValue = data["drawflow"]["Home"]["data"][str(i)]["data"]["minvalue"]
                    maxValue = data["drawflow"]["Home"]["data"][str(i)]["data"]["maxvalue"]
                    
                    params = {
                        "MINVALUE": int(minValue),
                        "MAXVALUE": int(maxValue),
                    }
                    
                    node = Node(i, elementClass, params)
                    self.flow.AddNode(node)  

                elif elementClass == "addition":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)
                
                elif elementClass == "subtraction":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)
                
                elif elementClass == "multiplication":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)
                
                elif elementClass == "equal":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)
                
                elif elementClass == "notequal":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)
                
                elif elementClass == "greater":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)
                
                elif elementClass == "greaterequal":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)

                elif elementClass == "lower":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)

                elif elementClass == "lowerequal":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)
                
                elif elementClass == "and":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)
                
                elif elementClass == "or":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)
                
                elif elementClass == "division":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    connector2 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_2"]["connections"]
                    node = Node(i, elementClass, None)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    # Connector operator 2
                    nodeId2 = connector2[0]["node"]
                    nodeInp2 = connector2[0]["input"]
                    con2 = InputConnector(nodeId2, nodeInp2, ValueType.NUMERIC)
                    node.SetInputConnector(con2)

                    self.flow.AddNode(node)
                
                elif elementClass == "scale":
                    connector1 = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    minValue = data["drawflow"]["Home"]["data"][str(i)]["data"]["minvalue"]
                    maxValue = data["drawflow"]["Home"]["data"][str(i)]["data"]["maxvalue"]
                    
                    params = {
                        "MINVALUE": float(minValue),
                        "MAXVALUE": float(maxValue),
                    }

                    node = Node(i, elementClass, params)
                    # Connector operator 1
                    nodeId = connector1[0]["node"]
                    nodeInp = connector1[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)

                    self.flow.AddNode(node)

                elif elementClass == "model":

                    model_uuid = data["drawflow"]["Home"]["data"][str(i)]["data"]["model_uuid"]
                    features = int(data["drawflow"]["Home"]["data"][str(i)]["data"]["features_count"])

                    inputs = []
                    for x in range(1, features+1):
                        inputs.append(data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_" + str(x)]["connections"])

                    #print(inputs, file=sys.stderr)
                    #print(elementClass + " (UUID: " + model_uuid + ", Inputs: " + str(inputs) + ")", file=sys.stderr)
                    params = {
                        "UUID": model_uuid,
                        "FEATURES": features,
                        "MODEL": self.model,
                        "VARIABLES": self.variables,
                    }

                    node = Node(i, elementClass, params)
                    for z in range(0, features):
                        nodeId = inputs[z][0]["node"]
                        nodeInp = inputs[z][0]["input"]
                        con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                        node.SetInputConnector(con)
                        #print(z, file=sys.stderr)
                    self.flow.AddNode(node)

                elif elementClass == "chart":
                    connector = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    node = Node(i, elementClass, None)
                    nodeId = connector[0]["node"]
                    nodeInp = connector[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    self.flow.AddNode(node)
                
                elif elementClass == "display":
                    connector = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]
                    node = Node(i, elementClass, None)
                    nodeId = connector[0]["node"]
                    nodeInp = connector[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.NUMERIC)
                    node.SetInputConnector(con)
                    self.flow.AddNode(node)
                
                elif elementClass == "notification":
                    message = data["drawflow"]["Home"]["data"][str(i)]["data"]["message"]
                    connector = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]

                    params = {
                        "MESSAGE": message,
                    }

                    node = Node(i, elementClass, params)
                    nodeId = connector[0]["node"]
                    nodeInp = connector[0]["input"]
                    con = InputConnector(nodeId, nodeInp, ValueType.BOOLEAN)
                    node.SetInputConnector(con)
                    self.flow.AddNode(node)
                
                elif elementClass == "log":
                    connectors = data["drawflow"]["Home"]["data"][str(i)]["inputs"]["input_1"]["connections"]

                    # Need to check the total input connector and add them to node instance.
                    conCount = len(connectors)
                    node = Node(i, elementClass, None)

                    for pos in range(0,conCount):
                        node.SetInputConnector(InputConnector(connectors[pos]["node"], connectors[pos]["input"], ValueType.NUMERIC))
                    
                    self.flow.AddNode(node)

            except Exception as err:
                print("Error in element: " + str(err), file=sys.stderr)
                pass
        
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

class DetectionModel:

    def __init__(self):
        self.uuid = str(uuid.uuid4())
    
    def Setup(self, name, description, keywords, model, variables):
        self.name = name
        self.description = description
        self.keywords = keywords
        self.model = model
        self.datetime = datetime.now()
        self.variables = variables
        self.MSE = 1
        self.RMSE = 1
        self.R2 = 1
        self.modelType = "detection"
        self.dataStudio = DataStudio()
        self.realTestList = list()
        self.predTestList = list()
        self.predTestScore = list()
    
    def SetModelPath(self, path):
        self.modelPath = path
    
    def SetModelVersion(self, modelVersion, appVersion):
        self.modelVersion = modelVersion
        self.appversion = appVersion
    
    def SetModelType(self, modelType):
        self.modelType = modelType
    
    def SetTestData(self, realTest, predTest, predScore):
        self.realTestList = realTest
        self.predTestList = predTest
        self.predTestScore = predScore
          
    def SetDataStudioData(self, data):
        self.dataStudio = data
    
    def SetAutomationDiagram(self, diagramData):
        self.automationDiagram = diagramData
    
