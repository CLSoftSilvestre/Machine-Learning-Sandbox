
import paho.mqtt.client as mqtt
import time
from threading import Thread

class MqttTopic:
    def __init__(self, name, topic, qos):
        self.name = name
        self.topic = topic
        self.qos = qos
        self.payload = None

class MqttConnector:

    def __init__(self,  protocol='tcp', ip='127.0.0.1', port=1883):
        self.protocol = protocol
        self.ip = ip
        self.port = port
        self.topics = []
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        #self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.service = None
        self.stop = True
    
    def Connect(self):
        self.client.connect(self.ip, self.port, 60)
        # Subscribe to topics
        for topic in self.topics:
            self.client.subscribe(topic.topic, topic.qos)

    def Loop(self):
        self.stop = False
        while not self.stop and self.client.loop()==0:
            #print("Connected to Broker: " + str(self.client.is_connected()))
            pass

    def on_connect():
        pass

    def on_message(self, mosq, obj, msg):
        for topic in self.topics:
            if topic.topic == str(msg.topic):
                topic.payload = msg.payload.decode("utf-8")

    def StartService(self):
        self.service = Thread(target=self.Loop)
        self.service.start()

    def StopService(self):
        self.stop = True
        self.service.join()

