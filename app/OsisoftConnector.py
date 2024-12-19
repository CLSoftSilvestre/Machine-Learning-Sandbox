
import sys
import PIconnect as PI

class PiPoint:
    def __init__(self, name):
        self.name = name
        self.curValue = None

class OSIsoftConnector:
    def __init__(self):
        self.pipoints = []
        self.pointsNames = []

    def AddVariable(self, variable:PiPoint = 0):
        self.pipoints.append(variable)
        self.pointsNames.append(variable.name)
    
    def ReadVariables(self):
        try:
            with PI.PIServer() as server:
                points = server.search(self.pointsNames)

                for i in range(len(points)):
                    self.pipoints[i].curValue = points[i].current_value
            return None
        except Exception as err:
            return err
        
            

