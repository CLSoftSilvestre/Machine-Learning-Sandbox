import asyncio
from bleak import BleakClient, BleakScanner, uuids
from bleak.backends.characteristic import BleakGATTCharacteristic
import struct
from threading import Thread
import sys

# Alternative https://pypi.org/project/simplepyble/

class BLECharacteristics:
    def __init__(self, uuid, deviceId):
        self.uuid = uuid
        self.formatedUUID = uuids.normalize_uuid_16(int(uuid, 16))
        self.deviceId = deviceId
        self.description = uuids.uuidstr_to_str(self.uuid)
        self.value = None

class BLEConnector:

    def __init__(self, name, deviceId):
        self.name = name
        self.deviceId = deviceId
        self.characteristics = []
        self.service = None
        self.stop = True

    def notification_handler(self, characteristic: BleakGATTCharacteristic, data: bytearray):
        value = int.from_bytes(data, byteorder='little')
        # Update the value inf the correspondent Characteristic
        print(value, file=sys.stderr)
        for charac in self.characteristics:
            if charac.description == str(characteristic.description):
                charac.value = value
    
    async def readDevice(self):
        device = await BleakScanner.find_device_by_name(self.name, timeout=5.0)

        async with BleakClient(device) as client:
            for item in self.characteristics:
                data = await client.read_gatt_char(item.formatedUUID) 
                item.value = int.from_bytes(data, byteorder='little')


    