
import sys
import pandas as pd
if sys.platform == 'win32':
    import PIconnect as PI
    from PIconnect.PIConsts import SummaryType
    from PIconnect import _time

class PiPoint:
    def __init__(self, name, description = "", value = None, uom = None, calculation = None):
        self.name = name
        self.description = description
        self.curValue = value
        self.uom = uom
        
        # Parse the calculatioon type
        if calculation != None:
            if calculation == "total":
                self.calculation = SummaryType.TOTAL
            elif calculation == "minimum":
                self.calculation = SummaryType.MINIMUM
            elif calculation == "maximum":
                self.calculation = SummaryType.MAXIMUM
            elif calculation == "standard deviation":
                self.calculation = SummaryType.STD_DEV
            elif calculation == "range":
                self.calculation = SummaryType.RANGE
            elif calculation == "count":
                self.calculation = SummaryType.COUNT
            elif calculation == "average time-weighted":
                self.calculation = SummaryType.AVERAGE
            elif calculation == "average event-weighted":
                self.calculation = SummaryType.AVERAGE
            else:
                self.calculation = SummaryType.NONE

        else:
            self.calculation = None

class OSIsoftConnector:
    def __init__(self):
        self.pipoints = []
        self.pointsNames = []

    def AddVariable(self, variable:PiPoint = 0):
        self.pipoints.append(variable)
        self.pointsNames.append(variable.name)
    
    def ReadVariables(self):
        if sys.platform == 'win32':
            try:
                with PI.PIServer() as server:
                    points = server.search(self.pointsNames)

                    for i in range(len(points)):
                        self.pipoints[i].curValue = points[i].current_value
                return None
            except Exception as err:
                return err
        
def GetPiPointsList(prefix="506-UTILI-"):
    piPoints = []
    if sys.platform == 'win32':
        with PI.PIServer() as server:
            query = prefix + "*"
            pip = server.search(query)
            for piPoint in pip:
                piPoints.append(PiPoint(piPoint.name, piPoint.description, piPoint.current_value, piPoint.units_of_measurement))
    
    return piPoints  

def GetPiPointData(startTime, endTime, interval, piPoint:PiPoint=0):
    with PI.PIServer() as server:
        point = server.search(piPoint.name)[0]
        data = point.summaries(start_time=startTime, end_time=endTime, interval=interval, summary_types=piPoint.calculation)
        # Change the data column names
        data.columns = [piPoint.name]
        return data

def GetPiData(startTime, endTime, interval, piPoints):
    outData = pd.DataFrame()
    for point in piPoints:
        if point.name != "":
            tempData = GetPiPointData(startTime, endTime, interval, point)
            outData = pd.concat([outData, tempData], axis=1)
            #print("Pi Point : " + str(point.name) + ", Aggregation: " + str(point.calculation), file=sys.stderr)

    return outData


