
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
        self.uuid = str(uuid.uuid4())
        self.datetime = datetime.now()
        self.name = name
        self.password = password
        self.role = role

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

    def SetNodeRed(self, use, endpoint):
        self.NodeRed = use
        self.NodeRedEndpoint = endpoint
    
    def SetOllama(self, use, model, endpoint):
        self.Ollama = use
        self.OllamaModel = model
        self.OllamaEndpoint = endpoint
    
    def AddAppUser(self, user : AppUser = 0):
        self.users.append(user)
    
    def UserLogin(self, username, password):
        for user in self.users:
            if user.name == username and user.password == password:
                return user
            else:
                return None

