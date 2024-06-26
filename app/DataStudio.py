
from datetime import datetime
import uuid
import pandas as pd
import sys
from utils import CleanColumnHeaderName
from pandas.api.types import is_datetime64_any_dtype as is_datetime

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
        err = self.__performSingleOperation(operation)
        return err
    
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

    def EditOperation(self, uuid, params):
        # Check the operation that contains the same uuid and edit the params
        for operation in self.operations:
            if operation.uuid == uuid:
                operation.params = params
                self.error = False
                self.errorMessage = ""
                self.run = False
        # Perform all operation after editing.
        self.__performAllOperations()

    def __performSingleOperation(self, operation):
        # Clear nulls operation
        if operation.operation == "clearnull":
            try:
                self.processedData.dropna(how='any', axis=0, inplace=True)
                self.processedData.reset_index(drop=True)
                operation.run = True
            except Exception as error:
                operation.run = False
                operation.error = True
                operation.errorMessage = error
            return ""

        # Remove column
        elif operation.operation == "remcol":
            try:
                self.processedData = self.processedData.drop([operation.params], axis=1)
                operation.run = True
            except Exception as error:
                operation.run = False
                operation.error = True
                operation.errorMessage = error
            return ""
        
        # Filter rows
        elif(operation.operation == "filtercol"):

            try:
                column = operation.params[0]
                operator = operation.params[1]
                value = operation.params[2]

                tempFilteredDf = pd.DataFrame()
                tempFilteredDf = self.processedData

                if operator == "<":
                    tempFilteredDf = self.processedData.loc[self.processedData[column] < float(value)]
                elif operator == "<=":
                    tempFilteredDf = self.processedData.loc[self.processedData[column] <= float(value)]
                elif operator == ">":
                    tempFilteredDf = self.processedData.loc[self.processedData[column] > float(value)]
                elif operator == ">=":
                    tempFilteredDf = self.processedData.loc[self.processedData[column] >= float(value)]
                
                self.processedData = tempFilteredDf
                self.processedData.reset_index(drop=True, inplace=True)

                operation.run = True
            except Exception as error:
                operation.run = False
                operation.error = True
                operation.errorMessage = error
            return ""

        # Change column name
        elif(operation.operation == "setcolumnname"):
            try:
                column = operation.params[0]
                newname = operation.params[1]
                self.__changeColumnName(column, CleanColumnHeaderName(newname))
                operation.run = True
            except Exception as error:
                operation.run = False
                operation.error = True
                operation.errorMessage = error

            return "Variable [" + str(column) + "] name changed to [" + str(newname) + "]"

        # Change column data type
        elif(operation.operation == "setdatatype"):
            try:
                column = operation.params[0]
                datatype = operation.params[1]
                err = self.__changeDatatype(column, datatype)
                print("3 - erro na change data type :" + err, file=sys.stderr)
                operation.run = True
            except Exception as error:
                operation.run = False
                operation.error = True
                operation.errorMessage = error
            if err != "":
                operation.run = False
                operation.error = True
                operation.errorMessage = err
            return err

        # Apply script to variable
        elif(operation.operation == "script"):
            try:
                # Change the variable name
                script = operation.params[0]
                context = operation.params[1]
                col = context.processedData
                exec(script)
                operation.run = True
                operation.error = False
                operation.errorMessage = ""
            except Exception as error:
                operation.run = False
                operation.error = True
                operation.errorMessage = error
                return "ERROR: Applying script. " + str(error)
            return ""

    def __performAllOperations(self):
        self.processedData = self.rawData.copy()
        # perform all operation in a single pass
        for operation in self.operations:
            self.error = False
            self.errorMessage = ""
            self.run = False
            self.__performSingleOperation(operation)

    def __changeDatatype(self, column, datatype):
        try:
            if (datatype == datetime):
                self.processedData[column] = pd.to_datetime(self.processedData[column], dayfirst=True)
                # self.processedData[column] = pd.to_numeric(self.processedData[column], downcast='float')
            else:
                self.processedData[column] = self.processedData[column].astype(datatype)
        except Exception as error:
            #print("ERROR: Updating data type of column ", column, " to .", error, datatype, file=sys.stderr)
            return "ERROR: Updating data type of column, " + str(error)
        return ""
    
    def __changeColumnName(self, column, newname):
        self.processedData.rename(columns={column:newname}, inplace=True)

    # Operations to use on script
    def AddColumn(self, baseColumn, name):
        self.processedData[name] = self.processedData[baseColumn]
    
    def ReplaceNaN(self, column , value):
        self.processedData[column] = self.processedData[column].fillna(value, inplace=True)
    
    def ReplaceNaN_Average(self, column):
        x = self.processedData[column].mean()
        self.processedData[column] = self.processedData[column].fillna(x, inplace=True)
    
    def ReplaceNaN_Median(self, column):
        x = self.processedData[column].median()
        self.processedData[column] = self.processedData[column].fillna(x, inplace=True)
    
    def ReplaceNaN_Mode(self, column):
        x = self.processedData[column].mode()
        self.processedData[column] = self.processedData[column].fillna(x, inplace=True)

    def RemoveDuplicates(self):
        self.processedData.drop_duplicates(inplace=True)

    def SetColumnName(self, column, name):
        self.__changeColumnName(column, CleanColumnHeaderName(name))

    def ExpandCategories(self, column):
        dummies = pd.get_dummies(self.processedData[column])
        if( dummies.shape[1] <= 10):
            self.processedData = pd.concat([self.processedData, dummies], axis="columns")
        else:
            raise Exception("More than 10 categories detected.")

    def AggWeek(self, dateColumn, mode):
        if (is_datetime(self.processedData[dateColumn])):
            try:
                if( mode == 'sum'):
                    self.processedData= self.processedData.groupby([self.processedData[dateColumn].dt.year, self.processedData[dateColumn].dt.week], as_index=False).sum()
                elif (mode == 'mean'):
                    self.processedData= self.processedData.groupby([self.processedData[dateColumn].dt.year, self.processedData[dateColumn].dt.week], as_index=False).mean()
            except:
                print("ERROR")
        else:
            raise Exception("Column " + dateColumn + " is not datetime format.")
    
    def AggMonth(self, dateColumn, mode):
        if (is_datetime(self.processedData[dateColumn])):
            try:
                if( mode == 'sum'):
                    self.processedData= self.processedData.groupby([self.processedData[dateColumn].dt.year, self.processedData[dateColumn].dt.month], as_index=False).sum()
                elif (mode == 'mean'):
                    self.processedData= self.processedData.groupby([self.processedData[dateColumn].dt.year, self.processedData[dateColumn].dt.month], as_index=False).mean()
            except:
                print("ERROR")
        else:
            raise Exception("Column " + dateColumn + " is not datetime format.")

class DataOperation:
    def __init__(self, operation, params):
        self.uuid = str(uuid.uuid4())
        self.operation = operation
        self.params = params
        self.error = False
        self.errorMessage = ""
        self.run = False
