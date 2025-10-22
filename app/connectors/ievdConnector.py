import requests
from threading import Thread
import time

class EdgeAttribute:
    def __init__(self, anchor):
        self.anchor = anchor
        self.name = None
        self.curRawValue = None
        self.curProcValue = None

class EdgeConnector:

    def __init__(self, ip, username, password):
        self.ip = ip
        self.username = username
        self.password = password
        self.attributes = []

        # Auth info
        self.accessToken = ""
        self.expiresAt = 0
        self.refreshToken = ""

    def AddAttribute(self, attribute):
        self.attributes.append(attribute)

    def Connect(self):
        #autenticate in the server to retrieve the access token
        url ="https://" + self.ip + "/device/edge/api/v2/login/direct"

        body_data = {
            "username":self.username,
            "password":self.password
        }

        response = requests.post(url, json=body_data, verify=False)
        response_json = response.json()

        if response.status_code == 200:
            self.accessToken = response_json["accessToken"]
            self.expiresAt = response_json["expiresAt"]
            self.refreshToken = response_json["refreshToken"]
        else:
            print("EDGE: Login error!")

    def ReadAttributes(self):

        headerData = {
            "Cookie":"authToken="+self.accessToken
        }

        for attr in self.attributes:
            url = "https://" + self.ip + "/iih-essentials/DataService/anchor/v1/attributes/" + attr.anchor
            response = requests.get(url=url, headers=headerData, verify=False)
            #response_json = response.json()

            # Check if value is boolean
            if response.text == "true":
                attr.curRawValue = 1
            elif response.text == "false":
                attr.curRawValue = 0
            else:
                try:
                    attr.curRawValue = float(response.text)
                except ValueError:
                    attr.curRawValue = None
