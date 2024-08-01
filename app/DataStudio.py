
from datetime import datetime
import uuid
import pandas as pd
import sys
from utils import CleanColumnHeaderName
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.preprocessing import LabelEncoder

class DataStudio:

    def __init__(self):
        # Create new DataStudio Instance
        self.uuid = str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.operations = []
        self.rawData = pd.DataFrame()
        self.processedData = pd.DataFrame()
        self.console = []
    
    def Clear(self):
        self.rawData = pd.DataFrame()
        self.processedData = pd.DataFrame()
        self.operations = []
        self.__addToConsole("Data cleared.")
    
    def LoadData(self, rawData):
        self.rawData = rawData.copy()
        self.processedData = rawData.copy()
        countRows = len(self.processedData)
        countHeaders = len(self.processedData.axes[1])

        self.__addToConsole("Data imported with " + str(countHeaders) + " columns and " + str(countRows) + " rows.")

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
                before = len(self.processedData)
                self.processedData.dropna(how='any', axis=0, inplace=True)
                self.processedData.reset_index(drop=True)
                operation.run = True
                after = len(self.processedData)
                
                self.__addToConsole("Removed rows with NaN. " + str(before-after) + " rows affected.")
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

                before = len(self.processedData)

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

                after = len(self.processedData)

                operation.run = True

                self.__addToConsole("Filter " + column + " " + operator + " " + value + " applied. " + str(before-after) + " rows affected.")

            except Exception as error:
                operation.run = False
                operation.error = True
                operation.errorMessage = error
            return ""
        
        elif(operation.operation == "remoutliers"):
            try:
                column = operation.params[0]
                upper = operation.params[1]
                lower = operation.params[2]

                Q1 = self.processedData[column].quantile([0.25][0])
                Q3 = self.processedData[column].quantile([0.75][0])
                IQR = Q3-Q1

                wiskersMax = Q3 + 1.5 * IQR
                wiskersMin = Q1 - 1.5 * IQR

                #print("Q1 : " + str(Q1), file=sys.stderr)
                #print("Q3 : " + str(Q3), file=sys.stderr)
                #print("Max whisker : " + str(wiskersMax), file=sys.stderr)
                #print("Min whisker : " + str(wiskersMin), file=sys.stderr)

                if(upper == True):
                    tempFilteredDf = pd.DataFrame()
                    tempFilteredDf = self.processedData
                    before = len(self.processedData)

                    tempFilteredDf = self.processedData.loc[self.processedData[column] < wiskersMax]
                    self.processedData = tempFilteredDf
                    self.processedData.reset_index(drop=True, inplace=True)

                    after = len(self.processedData)
                    operation.run = True
                    self.__addToConsole("Removed upper outliers from " + column + ". " + str(before-after) + " rows affected.")

                if(lower == True):
                    tempFilteredDf = pd.DataFrame()
                    tempFilteredDf = self.processedData
                    before = len(self.processedData)

                    tempFilteredDf = self.processedData.loc[self.processedData[column] > wiskersMin]
                    self.processedData = tempFilteredDf
                    self.processedData.reset_index(drop=True, inplace=True)

                    after = len(self.processedData)
                    operation.run = True
                    self.__addToConsole("Removed lower outliers from " + column + ". " + str(before-after) + " rows affected.")

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

    def __addToConsole(self, information):
        now = datetime.now()
        nowStr = now.strftime('%d-%m-%Y %H:%M:%S')
        self.console.append(nowStr + " - " + information + '\n')
        #print(self.console, file=sys.stderr)

    # Operations to use on script
    def AddColumn(self, baseColumn, name):
        self.processedData[name] = self.processedData[baseColumn]
    
    def ReplaceNaN(self, column , value):
        before = len(self.processedData)
        self.processedData[column] = self.processedData[column].fillna(value, inplace=True)
        after = len(self.processedData)
        self.__addToConsole("Removed rows with NaN. " + str(before-after) + " rows affected.")
    
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
        before = len(self.processedData)
        self.processedData.drop_duplicates(inplace=True)
        after = len(self.processedData)
        self.__addToConsole("Removed duplicated values rows. " + str(before-after) + " rows affected.")

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

    def LabelEncoder(self, column):
        dummies = pd.get_dummies(self.processedData[column])
        if( dummies.shape[1] > 10):
            raise Exception("More than 10 categories detected.")
        else:
            le = LabelEncoder()
            le.fit(self.processedData[column])
            labelsList = ""
            labelsValues = ""

            for label in le.classes_:
                labelsList = labelsList + label + ", "
            
            for number in le.transform(le.classes_):
                labelsValues = labelsValues + str(number) + ", "
            

            self.processedData[column] = le.transform(self.processedData[column])
            
            self.__addToConsole("Column " + column + " labels [" + labelsList + "] encoded into [" + labelsValues + "].")

class DataOperation:
    def __init__(self, operation, params):
        self.uuid = str(uuid.uuid4())
        self.operation = operation
        self.params = params
        self.error = False
        self.errorMessage = ""
        self.run = False
