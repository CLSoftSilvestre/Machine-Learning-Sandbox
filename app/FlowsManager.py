import uuid
from datetime import datetime
from enum import Enum
from S7Connector import S7Connector, S7Variable
import sys
import time
from threading import Thread
import random
import pandas as pd


class NodeType(Enum):
    INBOUND = 1
    MODELS = 2
    OPERATION = 3
    OUTBOUND = 4

class ValueType(Enum):
    S7CONNECTION = 10
    MQTTCONNECTION = 20
    OPCUACONNECTION = 30
    NUMERIC = 40
    BOOLEAN = 50

class InputConnector():
    def __init__(self, nodeId, outputNumber, valueType:ValueType = 0):
        self.uuid = str(uuid.uuid4())
        # Definition of the source of data
        self.nodeId = nodeId
        self.outputNumber = outputNumber
        # Data on the input
        self.valueType = valueType
        self.value = None

class Flow():
    def __init__(self):
        self.uuid = str(uuid.uuid4())
        self.Nodes = []
        self.Connectors = []
        self.timestamp = datetime.now()
        self.s7plc = []
        self.s7variables = []
        self.stop = True
        self.service = None
    
    def AddNode(self, node):
        self.Nodes.append(node)
        return node
    
    def SetJsonLayout(self, layout):
        self.jsonLayout = layout
    
    def GetNodeById(self, id):
        for node in self.Nodes:
            if str(node.id) == str(id):
                return node
        print("Node " + str(id) + " not found!", file=sys.stderr)
    
    def Start(self):
        self.s7plc = []
        self.s7variables = []
        # Create list of variables before setting the Simatic connector
        for node in self.Nodes:
            if node.nodeClass == "s7variable":
                print("variable"+ str(int(node.params["DB"])) + " - " + str(int(node.params["START"])) + " - "+ str(int(node.params["SIZE"])))
                s7var = S7Variable("variable", int(node.params["DB"]), int(node.params["START"]), int(node.params["SIZE"]))
                node.rawObject.append(s7var)
                self.s7variables.append(s7var)

        # Setup the PLC connector
        for node in self.Nodes:
            if node.nodeClass == "s7connector":
                # Create Siemens Connector Instance
                s7con = S7Connector(ip=node.params["IP"], rack=int(node.params["RACK"]), slot=int(node.params["SLOT"]), tcp_port=102)
                for variable in self.s7variables:
                    s7con.AddVariable(variable)
                self.s7plc.append(s7con)
                s7con.Connect()
                print("Connected to Siemens PLC " + s7con.ip + " - Status: " + str(s7con.client.get_connected()))
                s7con.StartService()
        
        # Start the Loop of the Flow
        self.Restart()
    
    def __loop(self):
        self.stop = False
        while not self.stop:

            # First loop inputs
            for node in self.Nodes:
                if node.nodeClass == "s7variable":
                    node.clearError()
                    try:
                        #print("Node " + str(node.id) + " - Siemens variable value: " +str(node.rawObject[0].curProcValue), file=sys.stderr)
                        node.outputValue = float(node.rawObject[0].curProcValue)
                    except Exception as err:
                        node.setError(str(err))
                        #print("Error updating Node " + str(node.id) + " - value", file=sys.stderr)

                elif node.nodeClass == "static":
                    staticValue = node.params["STATICVALUE"]
                    node.outputValue = staticValue

                elif node.nodeClass == "random":
                    minValue = node.params["MINVALUE"]
                    maxValue = node.params["MAXVALUE"]
                    node.clearError()
                    print("Minvalue: " + str(minValue) + ", Maxvalue: " + str(maxValue))
                    try:
                        rndValue = random.randrange(minValue, maxValue)
                        node.outputValue = rndValue
                    except Exception as err:
                        node.setError(str(err))
                
            # Second loop operations
            for node in self.Nodes:
                if node.nodeClass == "addition":
                    node.clearError()
                    try:
                        # Get the input nodes
                        prevNodeId1 = node.inputConnectors[0].nodeId
                        prevNodeId2 = node.inputConnectors[1].nodeId
                        prevNode1 = self.GetNodeById(prevNodeId1)
                        prevNode2 = self.GetNodeById(prevNodeId2)

                        # Get the value of the input nodes
                        value1 = float(prevNode1.outputValue)
                        value2 = float(prevNode2.outputValue)
                        # Perform operation
                        node.outputValue = value1 + value2
                        #print("Addition performed: " + str(value1) + " + " + str(value2) + " = " + str(node.outputValue))
                    except Exception as err:
                        node.setError(str(err))
                        #print("Error updating addition operation " + str(err), file=sys.stderr)

                elif node.nodeClass == "subtraction":
                    node.clearError()
                    try:
                        # Get the input nodes
                        prevNodeId1 = node.inputConnectors[0].nodeId
                        prevNodeId2 = node.inputConnectors[1].nodeId
                        prevNode1 = self.GetNodeById(prevNodeId1)
                        prevNode2 = self.GetNodeById(prevNodeId2)

                        # Get the value of the input nodes
                        value1 = float(prevNode1.outputValue)
                        value2 = float(prevNode2.outputValue)
                        # Perform operation
                        node.outputValue = value1 - value2
                        #print("Addition performed: " + str(value1) + " - " + str(value2) + " = " + str(node.outputValue))
                    except Exception as err:
                        node.setError(str(err))
                        #print("Error updating subtraction operation " + str(err), file=sys.stderr)

                elif node.nodeClass == "multiplication":
                    node.clearError()
                    try:
                        # Get the input nodes
                        prevNodeId1 = node.inputConnectors[0].nodeId
                        prevNodeId2 = node.inputConnectors[1].nodeId
                        prevNode1 = self.GetNodeById(prevNodeId1)
                        prevNode2 = self.GetNodeById(prevNodeId2)

                        # Get the value of the input nodes
                        value1 = float(prevNode1.outputValue)
                        value2 = float(prevNode2.outputValue)
                        # Perform operation
                        node.outputValue = value1 * value2
                        #print("Addition performed: " + str(value1) + " * " + str(value2) + " = " + str(node.outputValue))
                    except Exception as err:
                        node.setError(str(err))
                        #print("Error updating multiplication operation " + str(err), file=sys.stderr)

                elif node.nodeClass == "division":
                    node.clearError()
                    try:
                        # Get the input nodes
                        prevNodeId1 = node.inputConnectors[0].nodeId
                        prevNodeId2 = node.inputConnectors[1].nodeId
                        prevNode1 = self.GetNodeById(prevNodeId1)
                        prevNode2 = self.GetNodeById(prevNodeId2)

                        # Get the value of the input nodes
                        value1 = float(prevNode1.outputValue)
                        value2 = float(prevNode2.outputValue)
                        # Perform operation
                        node.outputValue = value1 / value2
                        #print("Addition performed: " + str(value1) + " / " + str(value2) + " = " + str(node.outputValue))
                    except Exception as err:
                        node.setError(str(err))
                        #print("Error updating division operation " + str(err), file=sys.stderr)

            # Third loop model
            for node in self.Nodes:
                if node.nodeClass == "model":
                    node.clearError()
                    mlModel = node.params["MODEL"]
                    inpVariables = node.params["VARIABLES"]
                    featuresCount = node.params["FEATURES"]
                    inputsNodes=[]
                    inputData = pd.DataFrame()

                    for i in range(0, featuresCount):
                        inputsNodes.append(self.GetNodeById(node.inputConnectors[i].nodeId))

                    #print("Node " + str(node.id) + " - Siemens variable value: " +str(node.rawObject[0].curProcValue), file=sys.stderr)
                    for i, variable in enumerate(inpVariables):
                        print("Variable : " + str(variable.name) + ", Value : " + str(inputsNodes[i].outputValue), file=sys.stderr)
                        inputData[variable] = [float(inputsNodes[i].outputValue)]
                        #print("Variable : " + str(variable.name) + ", Value : " + str(inputsNodes[i].outputValue), file=sys.stderr)
                    
                    try:
                        result = mlModel.predict(inputData)
                    except Exception as err:
                        node.setError(str(err))
                    
                    if result:
                        try:
                            resultValue = result[0][0]
                        except:
                            resultValue = result[0]
                        node.outputValue = resultValue
                        #print("Prediction: " + str(resultValue), file=sys.stderr)
                    else:
                        node.outputValue = None
                        node.setError("Error predicting value.")

            # Forth loop outputs
            for node in self.Nodes:
                if node.nodeClass == "chart":
                    node.clearError()
                    try:
                        prevNodeId = node.inputConnectors[0].nodeId
                        #print("Chart data predious node id " + str(prevNodeId), file=sys.stderr)
                        prevNode = self.GetNodeById(prevNodeId)
                        value = prevNode.outputValue
                        node.innerStorageArray.append(value)
                        # Only stay with last 15 inputs...
                        if len(node.innerStorageArray) > 15:
                            node.innerStorageArray.pop(0)
                        node.outputValue = value
                        #print("Chart data previous node " + str(value), file=sys.stderr)
                        #print(node.innerStorageArray, file=sys.stderr)
                    except Exception as err:
                        node.setError(str(err))
                        #print("Error updating chart " + str(err), file=sys.stderr)
            
            time.sleep(10)
        return False
    
    def Stop(self):
        self.stop = True
        self.service.join()  
        self.s7plc = []
        self.s7variables = []
    
    def Restart(self):
        self.service=Thread(target=self.__loop)
        self.service.start()
        
class Node():
    def __init__(self, id, nodeClass, params):
        self.id = id
        self.nodeClass = nodeClass
        self.inputConnectors = []
        self.params = params
        self.outputValue = None
        self.innerStorageArray = []
        self.rawObject = []
        self.error = False
        self.errorText = ""
    
    def SetInputConnector(self, con:InputConnector=0):
        self.inputConnectors.append(con)
    
    def getInnerStorage(self):
        return self.innerStorageArray
    
    def clearError(self):
        self.error = False
        self.errorText = ""
    
    def setError(self, errText):
        self.error = True
        self.errorText = errText
        print("Error in node " + str(self.id) + ", class: " + str(self.nodeClass) + ", message: " + str(errText), file=sys.stderr)


    

