
import uuid
import pickle
import glob
from datetime import datetime

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

    def SetBase(self, useLogin):
        self.useLogin = useLogin

    def SetNodeRed(self, use, endpoint):
        self.NodeRed = use
        self.NodeRedEndpoint = endpoint
    
    def SetOllama(self, use, model, endpoint):
        self.Ollama = use
        self.OllamaModel = model
        self.OllamaEndpoint = endpoint