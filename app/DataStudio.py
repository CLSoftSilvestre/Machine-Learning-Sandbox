
from datetime import datetime
import uuid
import pandas as pd
import sys

class DataStudio:
    def __init__(self):
        # Create new DataStudio Instance
        self.uuid = str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.operations = []
        self.rawData = pd.DataFrame()
        self.processedData = pd.DataFrame()
    
    def Clear(self):
        self.rawData = pd.DataFrame()
        self.processedData = pd.DataFrame()
        self.operations = []
    
    def LoadData(self, rawData):
        self.rawData = rawData.copy()
        self.processedData = rawData.copy()

    def AddOperation(self, operation):
        self.operations.append(operation)
        self.__performSingleOperation(operation)
    
    def RemoveOperation(self, uuid):

        # Get the position to POP the element
        pos = 0
        i = -1
        for operation in self.operations:
            i +=1
            if operation.uuid == str(uuid):
                pos = i
        # print("Poping element UUID ", uuid, file=sys.stderr) 
        # print("Poping element at position ", pos, file=sys.stderr)  

        self.operations.pop(pos)
        self.__performAllOperations()
    
    def RemoveOperationId(self, id):
        self.operations.pop(int(id))
        self.__performAllOperations()
                
    def __performSingleOperation(self, operation):
        # Clear nulls operation
        if operation.operation == "clearnull":
            self.processedData.dropna(how='any', axis=0, inplace=True)
            self.processedData.reset_index(drop=True)

        # Remove column
        elif operation.operation == "remcol":
            self.processedData = self.processedData.drop([operation.params], axis=1)
        
        # Filter rows
        elif(operation.operation == "filtercol"):
            column = operation.params[0]
            min = operation.params[1]
            max = operation.params[2]

            tempFilteredDf = pd.DataFrame()
            tempFilteredDf = self.processedData

            if min != "":
                # print("Applying filtering column " + column + " with min value", file=sys.stderr)
                tempFilteredDf = self.processedData.loc[self.processedData[column] >= float(min)]
            
            if max != "":
                # print("Applying filtering column " + column + " with max value", file=sys.stderr)
                tempFilteredDf = self.processedData.loc[self.processedData[column] <= float(max)]
            
            self.processedData = tempFilteredDf

        # Change column name
        elif(operation.operation == "setcolumnname"):
            column = operation.params[0]
            newname = operation.params[1]
            self.__changeColumnName(column, newname)

        # Change column data type
        elif(operation.operation == "setdatatype"):
            column = operation.params[0]
            datatype = operation.params[1]
            self.__changeDatatype(column, datatype)

    def __performAllOperations(self):
        self.processedData = self.rawData.copy()
        # perform all operation in a single pass
        for operation in self.operations:
            self.__performSingleOperation(operation)

    def __changeDatatype(self, column, datatype):
        try:
            if (datatype == datetime):
                self.processedData[column] = pd.to_datetime(self.processedData[column], dayfirst=True)
            else:
                self.processedData[column] = self.processedData[column].astype(datatype)
        except Exception as error:
            print("ERROR: Updating data type of column ", column, " to .", error, datatype, file=sys.stderr)
            return False
        return True
    
    def __changeColumnName(self, column, newname):
        self.processedData.rename(columns={column:newname}, inplace=True)

class DataOperation:
    def __init__(self, operation, params):
        self.uuid = str(uuid.uuid4())
        self.operation = operation
        self.params = params
