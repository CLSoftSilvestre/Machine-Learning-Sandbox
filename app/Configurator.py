
import uuid
import pickle
import glob
from datetime import datetime
from enum import Enum
import bcrypt


class UserRole(Enum):
    ADMIN = 1
    AI_ENGINEER = 2
    DATA_ANALYST = 3
    DATA_CONSUMER = 4

class AppUser():
    def __init__(self, name, password, role : UserRole = 0):

        rawPassword = password.encode('ASCII')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(rawPassword, salt)

        self.uuid = str(uuid.uuid4())
        self.datetime = datetime.now()
        self.name = name
        self.password = hashed
        self.role = role
    
    def ChangeUserPassword(self, oldPassword, newPassword):

        if self.password == bcrypt.hashpw(oldPassword.encode('ASCII'), self.password):
            rawPassword = newPassword.encode('ASCII')
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(rawPassword, salt)
            self.password = hashed
            return True
        else:
            return False

class Configurator():

    def __init__(self):
        pass
    
    def SaveConfiguration(self, configuration, path):
        pickle.dump(configuration, open(path, 'wb'))
        return True
    
    def LoadConfiguration(self, path):
        loaded_config = pickle.load(open(path, 'rb'))
        return loaded_config
    
    def GetConfigFiles(self, path):
        res = []
        res = glob.glob(path)
        return res
    
    def GetConfigFilesList(self, path):
        configs = []
        files = self.GetConfigFiles(path)

        for file in files:
            configs.append(self.LoadConfiguration(file))
        
        return configs

class Configuration():

    def __init__(self):
        self.datatime = datetime.now()
        self.uuid = str(uuid.uuid4())
        self.users = []

    def SetBase(self, useLogin):
        self.useLogin = useLogin
    
    def SetPyGWalker(self, use):
        self.usePyGWalker = use

    def SetAutomation(self, use):
        self.enableAutomation = use

    def SetSiemensConnector(self, use):
        self.useSiemensConnector = use
    
    def SetMqttConnector(self, use):
        self.useMqttConnector = use
    
    def SetOpcUaConnector(self, use):
        self.useOpcUaConnector = use
    
    def AddAppUser(self, user : AppUser = 0):
        self.users.append(user)
    
    def UserLogin(self, username, password):
        for user in self.users:
            if user.name == username and user.password == bcrypt.hashpw(password.encode('ASCII'), user.password):
                return user
            else:
                return None

