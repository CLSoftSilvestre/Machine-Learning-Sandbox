
import time
from threading import Thread
from pyModbusTCP.client import ModbusClient

class ModbusHoldingRegister:
    def __init__(self, deviceId, address, size, register_type="holding"):
        self.deviceId = deviceId
        self.register_type = register_type
        self.address = address
        self.size = size
        self.curRawValue = None
        self.curProcValue = None


class ModbusConnector:
    def __init__(self, host="localhost", port=502, unit_id=1, auto_open=True, auto_close=True):
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.auto_open = auto_open
        self.auto_close = auto_close
        self.registers = []
        self.client = None
        self.service = None
        self.stop = True
        self.interval = 5

    def Connect(self):
        self.client = ModbusClient(self.host, self.port, self.unit_id, self.auto_open)
        return self.client.is_open

    def AddVariable(self, variable:ModbusHoldingRegister = 0):
        self.registers.append(variable)

    def ReadVariables(self):
        self.stop = False

        while not self.stop:
            if self.client.is_open:
                for var in self.registers:
                    if var.register_type == "holding":
                        var.curRawValue = self.client.read_holding_registers(var.address, var.size)
                    elif var.register_type == "coils":
                        var.curRawValue = self.client.read_coils(var.address, var.size)
                    elif var.register_type == "inputs":
                        var.curRawValue = self.client.read_input_registers(var.address, var.size)
                    # TODO: Parse the raw value to proc value
                
                time.sleep(self.interval)
            else:
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

