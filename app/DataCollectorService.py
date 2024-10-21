
import uuid
from FlowsManager import Flow
from FlowsManager import NodeType, ValueType, Flow, Node, InputConnector
import json
import sys

class DataCollectorService:

    def __init__(self):
        self.uuid = str(uuid.uuid4())

    def SetDiagram(self, diagramData):
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
                    field = data["drawflow"]["Home"]["data"][str(i)]["data"]["field"]
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

            except Exception as err:
                print("Error in element: " + str(err), file=sys.stderr)
                pass

    def StartService(self):
        if self.flow is not None:
            self.flow.Start()
    
    def StopService(self):
        if self.flow is not None:
            if self.flow.getStatus:
                print("Stopping DC service", file=sys.stderr)
                self.flow.Stop()
    