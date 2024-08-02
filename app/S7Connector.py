# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:57:06 2024

@author: CSILVESTRE
"""

import snap7 as s7
from snap7 import Client, Area, util
import time
from threading import Thread

class S7Variable:
    def __init__(self, name, db_number, start, size):
        self.name = name
        self.db_number = db_number
        self.start = start
        self.size = size
        self.curRawValue = None
        self.curProcValue = None

class S7Connector:
    
    def __init__(self, ip, rack, slot, tcp_port=102):
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.tcp_port = tcp_port
        self.variables = []
        self.client = s7.client.Client()
        self.service = None
        self.stop = True
        self.interval = 5
    
    def Connect(self):
        self.client.connect(self.ip, self.rack, self.slot, self.tcp_port)
        return self.client.get_connected()
    
    def AddVariable(self, s7Variable:S7Variable = 0):
        self.variables.append(s7Variable)
    
    def ReadVariables(self):
        self.stop = False
        
        while not self.stop:
            if self.client.get_connected():
                for var in self.variables:
                    var.curRawValue = self.client.db_read(var.db_number, var.start, var.size)
                    if var.size == 2:
                        var.curProcValue = util.get_int(var.curRawValue, 0)
                    elif var.size == 4:
                        var.curProcValue = util.get_real(var.curRawValue, 0)
                time.sleep(self.interval)
            # Stop service if not connected
            self.StopService()
        return False
    
    def StartService(self):
        self.service = Thread(target=self.ReadVariables)
        self.service.start()
    
    def StopService(self):
        self.stop = True
        self.service.join()
    
    def SetInterval(self, seconds):
        self.interval = seconds
