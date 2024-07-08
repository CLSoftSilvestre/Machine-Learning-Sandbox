# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:53:44 2023

@author: CSilvestre
"""

from flask import Flask, render_template, send_from_directory, request, redirect, url_for, jsonify, session, send_file
from flask_session import Session

from ModelManager import ModelManager
import pandas as pd
import sys
import math

from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
#from sklearn.inspection import permutation_importance

from sklearn.datasets import load_iris, load_diabetes, load_digits, load_wine, load_breast_cancer

from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif

from models import LinearRegression, KnnRegression, model_data_input, knn_regressor_params, base_params, svm_regressor_params, random_forest_regressor_params


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from PredictionModel import PredictionModel, InputFeature, ModelInformation, ReturnFeature, Prediction, PredictionBatch
from ModelManager import ModelManager
from utils import CleanColumnHeaderName

from Configurator import Configuration, Configurator, AppUser, UserRole

import os
import io
import json
from werkzeug.utils import secure_filename

# from outlierextractor import CreateOutliersBoxplot
from datetime import datetime

from DataStudio import DataStudio, DataOperation
import uuid
import copy

from urllib.request import urlopen
from urllib.error import *

# import for usage of LLM models (Ollama)
from chatAI import ChatAI

app = Flask(__name__, instance_relative_config=True)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['SECRET'] = "secret!123"
Session(app)

cfg = Configurator()
confList = []

mm = ModelManager()
modelsList = []
appversion = "1.3.10"
model_version = 6
nodeRedRunning = False
ollamaRunning = False

@app.context_processor
def inject_app_version():
    return dict(version=appversion)

@app.context_processor
def inject_model_version():
    return dict(mversion=model_version)

@app.context_processor
def set_global_html_variable_values():
    #session['autenticated'] = True
    if session.get('autenticated'):
        useruuid = session.get('useruuid')
        role = session.get('role')
        loggedIn = session.get('autenticated')
        name = session.get('user')
    else:
        role = ""
        loggedIn = False
        name = ""
        useruuid = ""

    template_config = {'loggedin': loggedIn, 'username': name, 'role': role, 'useruuid':useruuid}

    return template_config

@app.context_processor
def show_session_warning():
    if session.get('warning') == None:
        session['warning'] = ""
    if session.get('information') == None:
        session['information'] = ""

    warning = ""
    info = ""
    warning = session.get('warning')
    info = session.get("information")
    template_config = {'warning_text': warning, 'info_text':info}
    session['warning'] = ""
    session['information'] = ""
    return template_config

@app.context_processor
def show_app_configurations():
    config = Configuration()
    if len(confList) > 0:
        config = confList[0]

    return dict(app_config = config)

@app.route("/index")
@app.route("/")
def index():

    # Check if configuration exists inserver. Otherwise start new configuration process
    if(len(confList) == 0):
        return redirect('/configurator/')
    else:
        if (session.get('autenticated') == True):
            return redirect('/sandbox/')
        else:
            if confList[0].useLogin == True:
                #return redirect('/sandbox/')
                return render_template('index.html')
            else:
                session['user'] = "Local User"
                session['useruuid'] = uuid.uuid4()
                session['role'] = UserRole.ADMIN
                session['autenticated'] = True
                return redirect('/sandbox/')

@app.route("/configurator/", methods=['GET', 'POST'])
def configurator():
    if(request.method == 'POST'):

        useNodeRed = bool(request.form.get('useNodeRed'))
        nodeRedPath = request.form['nodeRedEndpoint']
        useOllama = bool(request.form.get('useOllama'))
        ollamaPath = request.form['ollamaEndpoint']
        ollamaModel = request.form['ollamaModel']

        configuration = Configuration()

        # Check if configuration already exists
        if len(confList)>0:
            useAuth = confList[0].useLogin
            users = confList[0].users
            configuration.users = users

        else:
            # Get data and save in configuration file
            useAuth = bool(request.form.get('useAutentication'))
            adminPassword = request.form['adminPassword']

            #Add base admin user
            admUser = AppUser("admin", adminPassword, UserRole.ADMIN)
            configuration.AddAppUser(admUser)
        
        configuration.SetBase(useAuth)
        configuration.SetNodeRed(useNodeRed, nodeRedPath)
        configuration.SetOllama(useOllama, ollamaModel, ollamaPath)

        cfMan = Configurator()
        configName = "base.conf"
        filepath = os.path.join(app.root_path, 'config', configName)

        cfMan.SaveConfiguration(configuration, filepath)

        UpdateConfigurationList()

        return redirect('/index')

    else:
        if len(confList)==0: 
            return render_template('configurator.html')
        elif len(confList)>0 and session.get('role') == UserRole.ADMIN:
            return render_template('configurator.html')
        else:
            return redirect('/index')

@app.route("/changepassword/", methods=['POST'])
def changepassword():
    if(request.method == 'POST'):
        useruuid = request.form['useruuid']
        oldPassword = request.form['oldpassword']
        newPassword = request.form['newpassword']
        repeatPassword = request.form['repeatpassword']

        if(repeatPassword == newPassword):
            # print("Passwords são iguais", file=sys.stderr)
            for user in confList[0].users:
                if user.uuid == useruuid:
                    # print("Utilizador encontrado", file=sys.stderr)
                    success = user.ChangeUserPassword(oldPassword, newPassword)

                    if (success == True):
                        
                        # Save configuration file with user info
                        configName = "base.conf"
                        filepath = os.path.join(app.root_path, 'config', configName)
                        cfg.SaveConfiguration(confList[0], filepath)

                        session['information'] = "Password changed with success!"
                        return redirect('/index')
                    else:
                        session['warning'] = "Error changing password! Please check old password."
                        return redirect('/index')

        session['warning'] = "Error changing password! New password does not match."
        return redirect('/index')

    return redirect('/index')

@app.route("/sandbox/", methods=['GET'])
def sandbox():
    if (session.get('autenticated') != True):
        return redirect('/index')
    
    # Load the models in the first usage
    if len(modelsList) == 0:
        UpdateModelsList()
    
    if len(confList) == 0:
        UpdateConfigurationList()
    
    if len(confList) > 0:
        conf = confList[0]

    return render_template('sandbox.html', models=modelsList, config = conf)

@app.route("/usage/", methods=['GET'])
def usage():

    return render_template('usage.html')

@app.route("/flows/", methods=['GET'])
def flows():
    conf = confList[0]
    return render_template('flows.html', nodeRed = nodeRedRunning, config = conf)

@app.route("/details/<uuid>", methods=['GET'])
def details(uuid):
    for model in modelsList:
        if (model.uuid == uuid):
            equation = ""
            # Check if it's linear regression and calculate the equation
            if str(model.model) == "LinearRegression()":
                equation = "y = "
                pos = 0
                for variable in model.model.coef_:
                    equation = equation + "( " + str(round(variable,3)) + " x " + model.variables[pos].name + " ) + "
                    pos = pos+1
                
                equation = equation + str(round(model.model.intercept_,3))


            return render_template('details.html', Model=model, equation=equation)
  
    return render_template('details.html')

@app.route("/download/<uuid>", methods=['GET'])
def download(uuid):

    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    for model in modelsList:
        if (model.uuid == uuid):
            return render_template('download.html', Model=model)
        
    return render_template('download.html')

@app.route("/downloaddatastudio/", methods=['GET'])
def downloaddatastudio():
    buffer = io.BytesIO()
    session['data_studio'].processedData.to_csv(buffer, sep=';', decimal=',', index=False)
    buffer.seek(0)

    return send_file(buffer, download_name='dataset.csv', mimetype='text/csv')

@app.route("/downloadmodel/<uuid>", methods=['GET'])
def downloadmodel(uuid):

    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    for model in modelsList:
        if (model.uuid == uuid):
            modelName = model.modelPath.split("\\")[-1]
            try:
                modelsPath = os.path.join(app.root_path, 'models')
                return send_from_directory(modelsPath,modelName, as_attachment=True)
            except:
                modelsPath = os.path.join(os.getcwd(), 'models')
                return send_from_directory(modelsPath,modelName, as_attachment=True)
            
    return render_template('download.html')

@app.route("/predict/<uuid>", methods=['GET', 'POST'])
def predict(uuid):
    
    if (request.method == 'GET'):
        for model in modelsList:
            if (model.uuid == uuid):
                return render_template('predict.html', Model=model)
        return render_template('predict.html') 
    
    if(request.method == 'POST'):

        inputData = pd.DataFrame()
        features = []

        for key in request.form:
            if (len(request.form[key])==0):
                # Is empty and should be number
                return redirect(url_for('index'))
            
            inputData[key]=[request.form[key]]
            features.append(ReturnFeature(key, float(inputData[key])))

        print(inputData, file=sys.stderr)

        # Check the model
        for model in modelsList:
            if (model.uuid == uuid):
                activeModel = model
                try:
                    result =model.model.predict(inputData)
                except:
                    return render_template('predict.html', Model=activeModel, Error=True) 

        #print(round(result[0][0],2), file=sys.stderr)

        if result:
            try:
                resultValue = result[0][0]
            except:
                resultValue = result[0]
            #return render_template('predict.html', Model=activeModel, Result="{:,}".format(round(result[0][0],2)))
            # Calculate the min and max expected according MSE score
            minResult = resultValue - (math.sqrt(activeModel.MSE)/2)
            maxResult = resultValue + (math.sqrt(activeModel.MSE)/2)

            return render_template('predict.html', Model=activeModel, Result="{:,}".format(round(resultValue,2)), Features=features, MinResult="{:,}".format(round(minResult,2)), MaxResult="{:,}".format(round(maxResult,2)), Units=activeModel.pVariableUnits) 
        else:
            return redirect(url_for('index'))

@app.route("/refresh/", methods=['GET'])
def refresh():
    UpdateModelsList()
    return redirect(url_for('index'))

@app.route("/train/", methods=['GET', 'POST'])
def train():

    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    # acording to the command will perform mod on the data before push again to the page.

    if(request.method == 'POST'):
        # print(request.form['mod'], file=sys.stderr)
        # print(type(request.form['mod']), file=sys.stderr)s
        
        if(request.form['mod']=="setdependent"):
            session['temp_df_y'] = session['temp_df'][request.form['column']]
            session['temp_df_y_name'] = request.form['column']
            session['temp_df_x'] = session['temp_df'].loc[:,session['temp_df'].columns != session['temp_df_y_name']]

            if len(session['temp_df_units']) > 0:
                for unit in session['temp_df_units']:
                    if unit[0] == session['temp_df_y_name']:
                        session['temp_variable_units'] = unit[1]

        elif(request.form['mod']=="setproperties"):
            propList = []
            session['temp_df_units'] = []
            session['temp_variable_units'] = ""

            for i in session['temp_df']:
                varUnit = request.form[i]
                print("Objecto " + i + " - Unidade: " + varUnit, file=sys.stderr)
                propList.append((i, varUnit))
                session['temp_df_units'] = propList
            
            if session['temp_df_y_name'] != "":
                session['temp_variable_units'] = request.form[session['temp_df_y_name']]

    if session['temp_df'].columns.size > 0:
        return render_template('train.html', titles=session['temp_df'].columns.values, uploaded=True, dependend = session['temp_df_y_name'], units=session['temp_df_units'])
    else:
        emptyList = []
        emptyList.append((0,1))
        return render_template('train.html', rawdata=emptyList)

@app.route("/uploader/", methods=['GET', 'POST'])
def uploader():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    try:

        if request.method == 'POST':
            f = request.files['file']
            if f:
                # Process the file
                sep = request.form['sep']
                dec = request.form['dec']

                session['temp_df'] = pd.read_csv(f,sep=sep, decimal=dec)

                # Clean the names of the headers to avid problems in fitration and deletion
                oldNames = session['temp_df'].columns.tolist()
                newNames = list()

                for i in range(len(oldNames)):
                    newNames.append(CleanColumnHeaderName(oldNames[i]))
                
                session['temp_df'].columns = newNames

                # Start a new DataStudio Session
                session['data_studio'] = DataStudio()
                tempData = session['temp_df'].copy()
                session['data_studio'].LoadData(tempData)


                return redirect('/datastudio')
            else:
                return redirect('/datastudio')
        else:
            return redirect('/datastudio')
    
    except Exception as error:
        session['warning'] = "Error: " + str(error)
        return redirect('/datastudio')

@app.route("/cleardataset/", methods=['GET'])
def cleardataset():
    session['temp_df'] = pd.DataFrame()
    session['temp_df_y'] = pd.DataFrame()
    session['temp_df_y_name'] = ""
    session['temp_df_x'] = pd.DataFrame()
    session['heatmap_base64_jpgData'] = ""
    session['outliers_base64_jpgData'] = ""
    session['data_studio'] = DataStudio()

    return render_template('datastudio.html')

@app.route("/datastudio/", methods=['GET', 'POST'])
def datastudio():

    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    # Check if user configured the OLLAMA connection
    #config = Configuration()
    config = confList[0]

    
    # acording to the command will perform mod on the data before push again to the page.

    if(request.method == 'POST'):
        if (request.form['mod']=="clearnull"):
            # DataStudio Session update
            dataOperation = DataOperation("clearnull", None)
            session['data_studio'].AddOperation(dataOperation)

        elif (request.form['mod']=="remcol"):
            # DataStudio Session update
            dataOperation = DataOperation("remcol", request.form['column'])
            session['data_studio'].AddOperation(dataOperation)

        elif(request.form['mod']=="filtercol"):

            # min = request.form['minimum']
            # max = request.form['maximum']
            column = request.form['column']
            operator = request.form['operatortype']
            value = request.form['filtervalue']
   
            # DataStudio Session update
            params = [column, operator, value]
            dataOperation = DataOperation("filtercol", params)
            session['data_studio'].AddOperation(dataOperation)
        
        elif(request.form['mod']=='remoutliers'):
            column = request.form['column']
            top = bool(request.form.get('upperOutliers'))
            bottom = bool(request.form.get('lowerOutliers'))

            params = [column, top, bottom]
            dataOperation = DataOperation("remoutliers", params)
            session['data_studio'].AddOperation(dataOperation)
        
        elif(request.form['mod']=="deloperation"):
            operationUUID = request.form['uuid']
            id = request.form['pos']
            session['data_studio'].RemoveOperationId(id)

        elif(request.form['mod']=="setdatatype"):
            columnName = request.form['column']
            dataType = request.form['coldatatype']

            # Change the datatype according to the new type provided
            if(dataType == "string"):
                newDatatype = str
            elif(dataType == "datetime"):
                newDatatype = datetime
            elif(dataType == "int"):
                newDatatype = int
            elif(dataType == "float"):
                newDatatype = float
            else:
                newDatatype = object

            params=[columnName, newDatatype]
            dataOperation = DataOperation("setdatatype", params)
            err = session['data_studio'].AddOperation(dataOperation)
            print("1 - erro na add operation :" + err, file=sys.stderr)
            session['warning'] = err

        elif(request.form['mod']=="setname"):
            columnName = request.form['column']
            newName = request.form['newname']

            # Change columns name if the one provided is not equal
            if(columnName != newName):
                params = [columnName, newName]
                dataOperation = DataOperation("setcolumnname", params)
                info = session['data_studio'].AddOperation(dataOperation)
                #session['information'] = info
        
        elif(request.form['mod']=="usedataset"):
            # Reset variables before
            session['temp_df'] = pd.DataFrame()
            session['temp_df_y'] = pd.DataFrame()
            session['temp_df_y_name'] = ""
            session['temp_df_x'] = pd.DataFrame()
            # session['heatmap_base64_jpgData'] = ""
            # session['outliers_base64_jpgData'] = ""
            session['temp_df_units'] = []
            session['temp_best_models'] = []
            session['temp_df'] = session['data_studio'].processedData.copy()

            return redirect('/train')
        
        elif(request.form['mod']=="script"):
            script = request.form['code']

            # Apply script on variable
            params = [script, session['data_studio']]
            dataOperation = DataOperation("script", params)
            err = session['data_studio'].AddOperation(dataOperation)
            session['warning'] = err
 
        elif(request.form['mod']=="editoperation"):
            operationType = request.form['type']
            operationuuid = request.form['uuid']

            # Check with kind of operation was edited:
            if(operationType == "setdatatype"):
                column = request.form['column']
                coldatatype = request.form['coldatatype']
                params = [column, coldatatype]
                session['data_studio'].EditOperation(operationuuid, params)
            
            elif(operationType == "remcol"):
                session['data_studio'].EditOperation(operationuuid, request.form['column'])
            
            elif(operationType == "setcolumnname"):
                column = request.form['column']
                newName = request.form['newname']
                params = [column, newName]
                session['data_studio'].EditOperation(operationuuid, params)

            elif(operationType == "filtercol"):
                column = request.form['column']
                operator = request.form['operatortype']
                value = request.form['filtervalue']
                params = [column, operator, value]
                session['data_studio'].EditOperation(operationuuid, params)
            
            elif(operationType == "remoutliers"):
                column = request.form['column']
                upper = bool(request.form.get('upperOutliers'))
                lower = bool(request.form.get('lowerOutliers'))
                params = [column, upper, lower]
                session['data_studio'].EditOperation(operationuuid, params)

            elif(operationType == "script"):
                script = request.form['editcode']
                params = [script, session['data_studio']]
                session['data_studio'].EditOperation(operationuuid, params)

    try:
        if session['data_studio'].processedData.columns.size > 0:
            # calculate correlation array
            matrix = []
            matrixTitles = []

            for titleX in session['data_studio'].processedData.columns.values:
                if(session['data_studio'].processedData[titleX].dtype.kind in 'iufc'):
                    matrixTitles.append(titleX)
                # X
                for titleY in session['data_studio'].processedData.columns.values:
                    if(session['data_studio'].processedData[titleX].dtype.kind in 'iufc' and session['data_studio'].processedData[titleY].dtype.kind in 'iufc'):
                        matrix.append({
                            "x":titleX, 
                            "y":titleY,
                            "v":session['data_studio'].processedData[titleX].corr(session['data_studio'].processedData[titleY])}
                            )

            print(config.Ollama, file=sys.stderr)


            return render_template('datastudio.html', tables=[session['data_studio'].processedData.head(n=10).to_html(classes='table table-hover table-sm text-center table-bordered', header="true")], titles=session['data_studio'].processedData.columns.values, uploaded=True, descTable=[session['data_studio'].processedData.describe().to_html(classes='table table-hover text-center table-bordered', header="true")], datatypes = session['data_studio'].processedData.dtypes, rawdata=list(session['data_studio'].processedData.values.tolist()), datastudio=session['data_studio'], matrixData = matrix, matrixTitles = matrixTitles, console=session['data_studio'].console, config=config)
        else:
            emptyList = []
            emptyList.append((0,1))
            return render_template('datastudio.html', rawdata=emptyList)
    except Exception as error:
        print(str(error), file=sys.stderr)

        #if error != 'data_studio':
            #session['warning'] = "Error: " + str(error)

        emptyList = []
        emptyList.append((0,1))
        return render_template('datastudio.html', rawdata=emptyList)

@app.route("/usermanager/", methods=['GET', 'POST'])
def usermanager():
    if (session.get('role') != 'Administrator'):
        return redirect('/notauthorized')

    return render_template('usermanager.html')

# Start of Models training
@app.route("/linear/", methods=['GET', 'POST'])
def linear():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    if(request.method == 'POST'):

        # Set the model
        #name = request.form['name']
        #description = request.form['description']
        #scaling = bool(request.form.get('scaling'))
        #featurered = bool(request.form.get('featurered'))
        #selectkbestk = int(request.form['selectkbestk'])
        #testsize = int(request.form['testsize']) / 100

        #keywords = request.form['keywords'].split(";")

        params = base_params()
        params.name = request.form['name']
        params.description = request.form['description']
        params.keywords = request.form['keywords'].split(";")
        params.scaling = bool(request.form.get('scaling'))
        params.featureRed = bool(request.form.get('featurered'))
        params.testSize = int(request.form['testsize']) / 100
        params.selectKBest = int(request.form['selectkbestk'])

        params.data.df_y = session['temp_df_y']
        params.data.df_y_name = session['temp_df_y_name']
        params.data.df_x = session['temp_df_x']

        pModel = LinearRegression(params)
        #pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)
        pModel.SetModelType("regression")
        pModel.SetPredictVariable(session['temp_df_y_name'], session['temp_variable_units'])

        pModel.SetDataStudioData(session['data_studio'])

        mMan = ModelManager()
        modelFileName = params.name + ".model"
        filepath = os.path.join(app.root_path, 'models', modelFileName)

        try:
            for feature in pModel.variables:
                for funit in session['temp_df_units']:
                    if feature.name == funit[0]:
                        feature.setUnit(funit[1])
        except:
            print("Error setting feature units.", file=sys.stderr)

        #print("filepath: ", filepath, file=sys.stderr)

        mMan.SaveModel(pModel, filepath)

        UpdateModelsList()

        return redirect('/index')
    
    else:
        featuresCount = len(session['temp_df_x'].columns)
        return render_template('linear.html', FeaturesCount = featuresCount)

@app.route("/knnreg/", methods=['GET', 'POST'])
def knnreg():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    if(request.method == 'POST'):

        # Set the model
        findbest = bool(request.form.get('findbest'))

        params = knn_regressor_params()
        params.name = request.form['name']
        params.description = request.form['description']
        params.keywords = request.form['keywords'].split(";")
        params.scaling = bool(request.form.get('scaling'))
        params.featureRed = bool(request.form.get('featurered'))
        params.testSize = int(request.form['testsize']) / 100
        params.selectKBest = int(request.form['selectkbestk'])

        params.n_neighbors = int(request.form['neighbors'])
        params.weights = request.form['weights']
        params.algorithm = request.form['algorithm']
        params.leaf_size = int(request.form['leaf'])
        params.p = int(request.form['p2'])
        params.metric = request.form['metric']
        params.metric_params = None

        if request.form['n_jobs'] == 'none':
            params.n_jobs = None
        else:
            params.n_jobs == -1

        #keywords = request.form['keywords'].split(";")

        session['temp_best_models'] = []
        bestModelsList = []
        if(findbest):
            for i in range(params.n_neighbors, params.n_neighbors+10, 1):
                pModel = KnnRegression(session['temp_df_y'], session['temp_df_y_name'], session['temp_df_x'], session['temp_df_units'], params)
                # Setup the remaing data of the model
                pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
                pModel.SetModelVersion(model_version, appversion)

                pModel.SetModelType("regression")
                pModel.SetPredictVariable(session['temp_df_y_name'], session['temp_variable_units'])
                pModel.SetDataStudioData(session['data_studio'])

                bestModelsList.append(pModel)
            session['temp_best_models'] = bestModelsList
            return redirect('/best')
        else:

            pModel = KnnRegression(session['temp_df_y'], session['temp_df_y_name'], session['temp_df_x'], session['temp_df_units'], params)
            # Setup the remaing data of the model
            #pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
            pModel.SetModelVersion(model_version, appversion)

            pModel.SetModelType("regression")
            pModel.SetPredictVariable(session['temp_df_y_name'], session['temp_variable_units'])

            pModel.SetDataStudioData(session['data_studio'])

            # Save the model
            mMan = ModelManager()
            modelFileName = params.name + ".model"
            filepath = os.path.join(app.root_path, 'models', modelFileName)
            mMan.SaveModel(pModel, filepath)

            UpdateModelsList()
            return redirect('/sandbox')
    
    else:
        # TODO: Add here the code to push the max of features to the page
        featuresCount = len(session['temp_df_x'].columns)
        return render_template('knnreg.html', FeaturesCount = featuresCount)

@app.route("/knn/", methods=['GET', 'POST'])
def knn():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    if(request.method == 'POST'):

        # Set the model
        n = int(request.form['neighbors'])
        weights = request.form['weights']
        algorithm = request.form['algorithm']
        leaf = int(request.form['leaf'])
        name = request.form['name']
        description = request.form['description']
        scaling = bool(request.form.get('scaling'))
        featurered = bool(request.form.get('featurered'))

        keywords = request.form['keywords'].split(";")

        if scaling:
            if featurered:
                knn = make_pipeline(StandardScaler(),SelectKBest(f_classif, k="all"), neighbors.KNeighborsClassifier(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf))
            else:
                knn = make_pipeline(StandardScaler(), neighbors.KNeighborsClassifier(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf))
        else:
            if featurered:
                knn = make_pipeline(SelectKBest(f_classif, k="all"), neighbors.KNeighborsClassifier(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf))
            else:
                knn = neighbors.KNeighborsClassifier(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf)
        
        # Set train/test groups
        x_train, x_test, y_train, y_test = train_test_split(session['temp_df_x'], session['temp_df_y'], test_size=0.33, random_state=42)

        # Train model
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        # Save model
        inputFeatures = []
        for item in session['temp_df_x']:
            inputFeatures.append(InputFeature(item, str(type(session['temp_df_x'][item][0])), "Description of " + item))
        
        # Set the feature unit
        try:
            for feature in inputFeatures:
                for funit in session['temp_df_units']:
                    if feature.name == funit[0]:
                        feature.setUnit(funit[1])
        except:
            print("Error setting feature units.", file=sys.stderr)
        
        desc = pd.DataFrame(session['temp_df_x'])

        for i in range(len(inputFeatures)):
            featureName = inputFeatures[i].name
            inputFeatures[i].setDescribe(desc[featureName].describe())

        pModel = PredictionModel()
        pModel.Setup(name,description,keywords, knn, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))
        pModel.SetTestData(y_test, y_pred)
        pModel.SetModelVersion(model_version, appversion)
        pModel.SetModelType("classification")
        pModel.SetPredictVariable(session['temp_df_y_name'], session['temp_variable_units'])

        pModel.SetDataStudioData(session['data_studio'])

        mMan = ModelManager()
        modelFileName = name + ".model"
        filepath = os.path.join(app.root_path, 'models', modelFileName)

        print("filepath: ", filepath, file=sys.stderr)

        mMan.SaveModel(pModel, filepath)

        UpdateModelsList()

        return redirect('/index')
    
    else:
        return render_template('knn.html')

@app.route("/randomforest/", methods=['GET', 'POST'])
def randomforest():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    if(request.method == 'POST'):

        # Set the model
        maxdepth = int(request.form['maxdepth'])
        randomstate = int(request.form['randomstate'])
        name = request.form['name']
        description = request.form['description']
        scaling = bool(request.form.get('scaling'))
        featurered = bool(request.form.get('featurered'))
        keywords = request.form['keywords'].split(";")

        if scaling:
            if featurered:
                clf = make_pipeline(StandardScaler(),SelectKBest(f_classif, k="all"), RandomForestClassifier(max_depth=maxdepth, random_state=randomstate))
            else:
                clf = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=maxdepth, random_state=randomstate))
        else:
            if featurered:
                clf = make_pipeline(SelectKBest(f_classif, k="all"), RandomForestClassifier(max_depth=maxdepth, random_state=randomstate))
            else:
                clf = RandomForestClassifier(max_depth=maxdepth, random_state=randomstate)
        
        # Set train/test groups
        x_train, x_test, y_train, y_test = train_test_split(session['temp_df_x'], session['temp_df_y'], test_size=0.33, random_state=42)

        # Train model
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)


        # Save model
        inputFeatures = []
        for item in session['temp_df_x']:
            inputFeatures.append(InputFeature(item, str(type(session['temp_df_x'][item][0])), "Description of " + item))

        # Set the feature unit
        try:
            for feature in inputFeatures:
                for funit in session['temp_df_units']:
                    if feature.name == funit[0]:
                        feature.setUnit(funit[1])
        except Exception as error:
            session['warning'] = "Error: " + str(error)
            print("Error setting feature units.", file=sys.stderr)

        desc = pd.DataFrame(session['temp_df_x'])

        for i in range(len(inputFeatures)):
            featureName = inputFeatures[i].name
            inputFeatures[i].setDescribe(desc[featureName].describe())
        
        pModel = PredictionModel()
        pModel.Setup(name,description,keywords,clf, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))
        pModel.SetTestData(y_test, y_pred)
        pModel.SetModelVersion(model_version, appversion)
        pModel.SetModelType("classification")
        pModel.SetPredictVariable(session['temp_df_y_name'], session['temp_variable_units'])

        pModel.SetDataStudioData(session['data_studio'])

        mMan = ModelManager()
        modelFileName = name + ".model"
        filepath = os.path.join(app.root_path, 'models', modelFileName)

        print("filepath: ", filepath, file=sys.stderr)

        mMan.SaveModel(pModel, filepath)

        UpdateModelsList()

        return redirect('/index')
    
    else:
        return render_template('randomforest.html')

@app.route("/randomforestreg/", methods=['GET', 'POST'])
def randomforestreg():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    if(request.method == 'POST'):

        # Set the model
        estimators = int(request.form['estimators'])
        maxdepth = int(request.form['maxdepth'])
        randomstate = int(request.form['randomstate'])
        name = request.form['name']
        description = request.form['description']
        scaling = bool(request.form.get('scaling'))
        featurered = bool(request.form.get('featurered'))
        selectkbestk = int(request.form['selectkbestk'])
        keywords = request.form['keywords'].split(";")

        if maxdepth == 0:
            maxdepth = None

        if scaling:
            if featurered:
                clf = make_pipeline(StandardScaler(),SelectKBest(f_classif, k=selectkbestk), RandomForestRegressor(n_estimators=estimators ,max_depth=maxdepth, random_state=randomstate))
            else:
                clf = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=estimators ,max_depth=maxdepth, random_state=randomstate))
        else:
            if featurered:
                clf = make_pipeline(SelectKBest(f_classif, k=selectkbestk), RandomForestRegressor(n_estimators=estimators ,max_depth=maxdepth, random_state=randomstate))
            else:
                clf = RandomForestRegressor(n_estimators=estimators ,max_depth=maxdepth, random_state=randomstate)
        
        # Set train/test groups
        x_train, x_test, y_train, y_test = train_test_split(session['temp_df_x'], session['temp_df_y'], test_size=0.33, random_state=42)

        # Train model
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)


        # Save model
        inputFeatures = []
        for item in session['temp_df_x']:
            inputFeatures.append(InputFeature(item, str(type(session['temp_df_x'][item][0])), "Description of " + item))

        # Set the feature unit
        try:
            for feature in inputFeatures:
                for funit in session['temp_df_units']:
                    if feature.name == funit[0]:
                        feature.setUnit(funit[1])
        except:
            print("Error setting feature units.", file=sys.stderr)

        
         # Calculate feature importances and update feature item.
        try:     
            importance = clf.feature_importances_
            for i, v in enumerate(importance):
                inputFeatures[i].setImportance(v)
        except:
            pass
        
        desc = pd.DataFrame(session['temp_df_x'])

        for i in range(len(inputFeatures)):
            featureName = inputFeatures[i].name
            inputFeatures[i].setDescribe(desc[featureName].describe())

        pModel = PredictionModel()
        pModel.Setup(name,description,keywords,clf, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))
        pModel.SetTestData(y_test, y_pred)
        #pModel.SetTrainImage(CreateImage(y_test, y_pred))
        #pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)
        pModel.SetModelType("regression")
        pModel.SetPredictVariable(session['temp_df_y_name'], session['temp_variable_units'])

        pModel.SetDataStudioData(session['data_studio'])

        mMan = ModelManager()
        modelFileName = name + ".model"
        filepath = os.path.join(app.root_path, 'models', modelFileName)

        print("filepath: ", filepath, file=sys.stderr)

        mMan.SaveModel(pModel, filepath)

        UpdateModelsList()

        return redirect('/index')
    
    else:
        featuresCount = len(session['temp_df_x'].columns)
        return render_template('randomforestreg.html', FeaturesCount = featuresCount)

@app.route("/svmreg/", methods=['GET', 'POST'])
def svmreg():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    if(request.method == 'POST'):

        # Set the model
        name = request.form['name']
        description = request.form['description']
        kernel = request.form['kernel']
        scaling = bool(request.form.get('scaling'))
        featurered = bool(request.form.get('featurered'))
        testsize = int(request.form['testsize']) / 100
        keywords = request.form['keywords'].split(";")

        if scaling == True:
            if featurered == True:
                clf = make_pipeline(StandardScaler(), SelectFromModel(SVR(kernel=kernel)), SVR(kernel=kernel))
            else:
                clf = make_pipeline(StandardScaler(), SVR(kernel=kernel))
        else:
            if featurered == True:
                clf = make_pipeline(SelectFromModel(SVR(kernel=kernel)), SVR(kernel=kernel))
            else:
                clf = SVR(kernel=kernel)
        
        # Set train/test groups
        x_train, x_test, y_train, y_test = train_test_split(session['temp_df_x'], session['temp_df_y'], test_size=testsize, random_state=42)

        # Train model
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        # Save model
        inputFeatures = []
        for item in session['temp_df_x']:
            inputFeatures.append(InputFeature(item, str(type(session['temp_df_x'][item][0])), "Description of " + item))

        # Set the feature unit
        try:
            for feature in inputFeatures:
                for funit in session['temp_df_units']:
                    if feature.name == funit[0]:
                        feature.setUnit(funit[1])
        except:
            print("Error setting feature units.", file=sys.stderr)

        desc = pd.DataFrame(session['temp_df_x'])

        for i in range(len(inputFeatures)):
            featureName = inputFeatures[i].name
            inputFeatures[i].setDescribe(desc[featureName].describe())

        pModel = PredictionModel()
        pModel.Setup(name,description,keywords,clf, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))
        pModel.SetTestData(y_test, y_pred)
        #pModel.SetTrainImage(CreateImage(y_test, y_pred))
        #pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)
        pModel.SetModelType("regression")
        pModel.SetPredictVariable(session['temp_df_y_name'], session['temp_variable_units'])

        pModel.SetDataStudioData(session['data_studio'])

        mMan = ModelManager()
        modelFileName = name + ".model"
        filepath = os.path.join(app.root_path, 'models', modelFileName)

        print("filepath: ", filepath, file=sys.stderr)

        mMan.SaveModel(pModel, filepath)

        UpdateModelsList()

        return redirect('/index')
    
    else:
        return render_template('svmreg.html')

@app.route("/treereg/", methods=['GET', 'POST'])
def treereg():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    if(request.method == 'POST'):

        # Set the model
        name = request.form['name']
        description = request.form['description']
        max_depth = int(request.form['maxdepth'])
        criterion = request.form['criterion']
        scaling = bool(request.form.get('scaling'))
        featurered = bool(request.form.get('featurered'))
        selectkbestk = int(request.form['selectkbestk'])
        testsize = int(request.form['testsize']) / 100
        keywords = request.form['keywords'].split(";")

        if scaling:
            if featurered:
                clf = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=selectkbestk), tree.DecisionTreeRegressor(max_depth=max_depth, criterion=criterion))
            else:
                clf = make_pipeline(StandardScaler(), tree.DecisionTreeRegressor(max_depth=max_depth, criterion=criterion))
        else:
            if featurered:
                clf = make_pipeline(SelectKBest(f_classif, k=selectkbestk), tree.DecisionTreeRegressor(max_depth=max_depth, criterion=criterion))
            else:
                clf = tree.DecisionTreeRegressor(max_depth=max_depth, criterion=criterion)

        # Set train/test groups
        x_train, x_test, y_train, y_test = train_test_split(session['temp_df_x'], session['temp_df_y'], test_size=testsize, random_state=42)

        # Train model
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        # Save model
        inputFeatures = []
        for item in session['temp_df_x']:
            inputFeatures.append(InputFeature(item, str(type(session['temp_df_x'][item][0])), "Description of " + item))
        
        # Set the feature unit
        try:
            for feature in inputFeatures:
                for funit in session['temp_df_units']:
                    if feature.name == funit[0]:
                        feature.setUnit(funit[1])
        except:
            print("Error setting feature units.", file=sys.stderr)
        
        # Calculate feature importances and update feature item.
        try:     
            importance = clf.feature_importances_
            for i, v in enumerate(importance):
                inputFeatures[i].setImportance(v)
        except:
            pass
        
        desc = pd.DataFrame(session['temp_df_x'])

        for i in range(len(inputFeatures)):
            featureName = inputFeatures[i].name
            inputFeatures[i].setDescribe(desc[featureName].describe())



        pModel = PredictionModel()
        pModel.Setup(name,description,keywords,clf, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))
        pModel.SetTestData(y_test, y_pred)
        #pModel.SetTrainImage(CreateImage(y_test, y_pred))
        #pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)
        pModel.SetModelType("regression")
        pModel.SetPredictVariable(session['temp_df_y_name'], session['temp_variable_units'])

        pModel.SetDataStudioData(session['data_studio'])

        mMan = ModelManager()
        modelFileName = name + ".model"
        filepath = os.path.join(app.root_path, 'models', modelFileName)

        print("filepath: ", filepath, file=sys.stderr)

        mMan.SaveModel(pModel, filepath)

        UpdateModelsList()

        return redirect('/index')
    
    else:
        featuresCount = len(session['temp_df_x'].columns)
        return render_template('treereg.html', FeaturesCount = featuresCount)

@app.route("/perceptronreg", methods=['GET', 'POST'])
def perceptronreg():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    if(request.method == 'POST'):
        # Set the model
        name = request.form['name']
        description = request.form['description']
        hiddenlayers = request.form['hiddenlayers']
        activation = request.form['activation']
        solver = request.form.get('solver')
        alfa = float(request.form.get('alfa'))
        learningrate = request.form.get('learningrate')
        learningrateinit = float(request.form.get('learningrateinit'))
        maxiter = int(request.form.get('maxiter'))
        keywords = request.form['keywords'].split(";")

        # convert hidden layers string to tupple
        try:
            convhiddenayers = eval(hiddenlayers)
        except:
            convhiddenayers = (100,)

        clf = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=convhiddenayers, random_state=1, max_iter=maxiter, activation=activation, solver=solver, alpha=alfa, learning_rate=learningrate, learning_rate_init=learningrateinit))

        # Set train/test groups
        x_train, x_test, y_train, y_test = train_test_split(session['temp_df_x'], session['temp_df_y'], test_size=0.33, random_state=42)

        # Train model
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        # Save model
        inputFeatures = []
        for item in session['temp_df_x']:
            inputFeatures.append(InputFeature(item, str(type(session['temp_df_x'][item][0])), "Description of " + item))
        
        # Set the feature unit
        try:
            for feature in inputFeatures:
                for funit in session['temp_df_units']:
                    if feature.name == funit[0]:
                        feature.setUnit(funit[1])
        except:
            print("Error setting feature units.", file=sys.stderr)

        desc = pd.DataFrame(session['temp_df_x'])

        for i in range(len(inputFeatures)):
            featureName = inputFeatures[i].name
            inputFeatures[i].setDescribe(desc[featureName].describe())

        pModel = PredictionModel()
        pModel.Setup(name,description,keywords,clf, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))
        pModel.SetTestData(y_test, y_pred)
        #pModel.SetTrainImage(CreateImage(y_test, y_pred))
        #pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)
        pModel.SetModelType("regression")
        pModel.SetPredictVariable(session['temp_df_y_name'], session['temp_variable_units'])

        pModel.SetDataStudioData(session['data_studio'])

        mMan = ModelManager()
        modelFileName = name + ".model"
        filepath = os.path.join(app.root_path, 'models', modelFileName)

        print("filepath: ", filepath, file=sys.stderr)

        mMan.SaveModel(pModel, filepath)

        UpdateModelsList()

        return redirect('/index')

    else:
        return render_template('perceptronreg.html')

@app.route("/usedataset/<uuid>", methods=['GET'])
def usedataset(uuid):

    if (session.get('autenticated') != True):
        return redirect('/notauthorized')

    for model in modelsList:
        if model.uuid == uuid:
            try:
                session['data_studio'] = copy.copy(model.dataStudio)
                session['data_studio'].console = []

            except Exception as error:
                session['warning'] = "Error copying Data Studio data. " + str(error)

    return redirect('/datastudio')

# End of Models training

@app.route("/best/", methods=['GET'])
def best():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    # Get the best models finded.
    return render_template('best.html', bestmodels=session['temp_best_models'])  

@app.route("/save/<uuid>", methods=['GET'])
def save(uuid):
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    for model in session['temp_best_models']:
        if (model.uuid == uuid):
            try:
                # Save the model
                mMan = ModelManager()
                modelFileName = model.name + ".model"
                filepath = os.path.join(app.root_path, 'models', modelFileName)
                mMan.SaveModel(model, filepath)

                UpdateModelsList()
                return redirect('/index')          
            except:
                print("Error saving model to " + model.modelPath, file=sys.stderr)
            UpdateModelsList()
    
    # Clean the temporary models list
    session['temp_best_models'] = []
            
    return redirect('/index')

@app.route("/batch/<uuid>", methods=['GET', 'POST'])
def batch(uuid):
    
    if(request.method == 'GET'):
        for model in modelsList:
            if (model.uuid == uuid):
                return render_template('batch.html', Model=model, Error=False, ErrorText="")
    
    if(request.method == 'POST'):
        dbPath = os.path.join(app.root_path, 'database', "mls.db")
        # 0 - Get the model
        for model in modelsList:
            if (model.uuid == uuid):
                tempModel = model

        # 1 - Check the file
        f = request.files['file']
        if f:
            # Process the file
            #sep = request.form['sep']
            #dec = request.form['dec']
            sep = ";"
            dec = ","
            session['temp_batch_df'] = []
            session['temp_predictions'] = []
            tempPrediction = []

            session['temp_batch_df'] = pd.read_csv(f,sep=sep, decimal=dec)

            # 1.1 - Check if headers are consistent with expected
            # check if it's the same number of header expected
            if len(session['temp_batch_df'].columns) == len(tempModel.variables):
                print("Header count match! ", file=sys.stderr)
            else:
                return render_template('batch.html', Error=True, ErrorText="Bad CSV format column headers provided the number of header dont match with the expected values!")
      
            # 1.2 - If we get here all headers exist lets calculate the predictions.   
            for index, row in session['temp_batch_df'].iterrows():

                try:
                    result = tempModel.model.predict([row])
                    try:
                        innerResult = result[0][0]
                    except:
                        innerResult = result[0]

                    tempPrediction.append(Prediction(innerResult, row))           
                    # add_Prediction(dbPath, datetime.now(), uuid, 1, 3)

                except Exception as error:
                    return render_template('batch.html', Error=True, ErrorText="Error predicting values! " + str(error))
                
                session['temp_predictions'] = PredictionBatch(tempModel, tempPrediction)

            return redirect('/batchresults')

        return render_template('batch.html', Error=True, ErrorText="Please select a valid CSV file!")
    
    return render_template('batch.html', Error=True, ErrorText="Model not found!")

@app.route("/batchresults/", methods=['GET'])
def batchresults():
    return render_template('results.html', Predictions=session['temp_predictions'])

@app.route("/delete/<uuid>", methods=['GET'])
def delete(uuid):
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    for model in modelsList:
        if (model.uuid == uuid):
            try:
                #print(model.name, file=sys.stderr)
                os.remove(model.modelPath)
            except:
                print("Error deleting model from " + model.modelPath, file=sys.stderr)
            UpdateModelsList()
            
    return redirect('/index')

@app.route("/import/", methods=['GET','POST'])
def importFile():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')

    if request.method == 'POST':
        f = request.files['file']
        if f:
            f.save(os.path.join(app.root_path, 'models', secure_filename(f.filename)))

            UpdateModelsList()      
            return redirect('/index')
        else:
            return render_template('import.html')
    else:
        return render_template('import.html')

def UpdateModelsList():
    global modelsList
    global mm
    modelspath = os.path.join(app.root_path, 'models', "*.model")
    modelsList = mm.GetModelsList(modelspath)
    #print(app.instance_path, file=sys.stderr)

def UpdateConfigurationList():
    global cfg
    global confList
    configPath = os.path.join(app.root_path, 'config', "*.conf")
    confList = cfg.GetConfigFilesList(configPath)

@app.route("/login/", methods=['POST'])
def Login():

    name = request.form['username']
    password = request.form['pswd']

    user = confList[0].UserLogin(name, password)

    print(user)

    if (user != None):
        session['user'] = user.name
        session['useruuid'] = user.uuid
        session['role'] = user.role
        session['autenticated'] = True

        # Train temp variables
        # Use session variables to store temp data
        # This will avoid setting data for other users
        session['temp_df'] = pd.DataFrame()
        session['temp_df_y'] = pd.DataFrame()
        session['temp_df_y_name'] = ""
        session['temp_df_x'] = pd.DataFrame()
        session['temp_df_units'] = []
        session['temp_best_models'] = []

        if len(confList) > 0:
            if confList[0].Ollama == True:
                session['ChatAI'] = ChatAI(confList[0].OllamaEndpoint, confList[0].OllamaModel)

    else:
        session['warning'] = "Error login. Wrong username or password!"
        session['autenticated'] = False

    return redirect('/index')

@app.route("/logout/", methods=['GET'])
def Logout():

    session.pop('user', None)
    session.pop('role', None)
    session.pop('autenticated', None)

    session['temp_df'] = pd.DataFrame()
    session['temp_df_y'] = pd.DataFrame()
    session['temp_df_y_name'] = ""
    session['temp_df_x'] = pd.DataFrame()
    session['heatmap_base64_jpgData'] = ""
    session['outliers_base64_jpgData'] = ""
    session['temp_df_units'] = []
    session['temp_best_models'] = []
    session.pop('data_studio', None)

    return redirect('/index')

@app.route("/notauthorized/",methods=['GET'])
def NotAuthorized():
    return render_template('notauthorized.html')

# API routes
@app.route("/api/GetModels", methods=['GET'])
def ApiGetModels():
    models = []

    for model in modelsList:
        models.append(ModelInformation(model.uuid, model.name, model.description))
    
    if(len(models) > 0):
        jsonStr = json.dumps([obj.__dict__ for obj in models])   
        return jsonStr, 200
        #return jsonify(jsonStr)
    else:
        return "No models found.", 404
    
@app.route("/api/GetModel/<uuid>", methods=['GET'])
def ApiGetModel(uuid):
    models = []

    for model in modelsList:
        if (model.uuid == uuid):
            models.append(ModelInformation(model.uuid, model.name, model.description))
            featuresJson = json.dumps(model.variables, default=InputFeature.serialize)
    
    if(len(models) > 0):

        data = {
                    "UUID" : models[0].uuid,
                    "Model" : models[0].name,
                    "Description" : models[0].description,
                    "Features": json.loads(featuresJson)
                }

        return jsonify(data)
    else:
        return "No models found.", 404

@app.route("/api/Predict/<uuid>", methods=['GET', 'POST'])
def ApiPredict(uuid):

    # return the list of input parameters for the model
    if (request.method == 'GET'):
        models = []

        for model in modelsList:
            if (model.uuid == uuid):
                models.append(ModelInformation(model.uuid, model.name, model.description))
                featuresJson = json.dumps(model.variables, default=InputFeature.serialize)
        
        if(len(models) > 0):

            data = {
                        "UUID" : models[0].uuid,
                        "Model" : models[0].name,
                        "Description" : models[0].description,
                        "Features": json.loads(featuresJson)
                    }

            return jsonify(data)
        else:
            return "No models found.", 404
    
    if(request.method == 'POST'):

        inputData = pd.DataFrame()
        features = []

        resultJson = json.loads(request.data)

        for arg in resultJson:
            inputData[arg]=[resultJson[arg]]
            features.append(ReturnFeature(arg, float(resultJson[arg])))
            featuresJson = json.dumps(features, default=ReturnFeature.serialize)

        # Check the model
        for model in modelsList:
            if (model.uuid == uuid):
                try:
                    result =model.model.predict(inputData)
                    
                    try:
                        innerResult = result[0][0]
                    except:
                        innerResult = result[0]
                    
                    minResult = innerResult - (math.sqrt(model.MSE)/2)
                    maxResult = innerResult + (math.sqrt(model.MSE)/2)

                    data = {
                        "UUID" : model.uuid,
                        "Model" : model.name,
                        "Description" : model.description,
                        "Prediction" : innerResult,
                        "MinPrediction" : minResult,
                        "MaxPrediction" : maxResult,
                        "Features": json.loads(featuresJson)
                    }

                    return jsonify(data), 200
                except:

                    return "Error predicting value.", 404
        
        return "Model not found", 404

# ChatAI API
@app.route("/api/chat/ask", methods=['POST'])
def AskQuestion():
    body = json.loads(request.data)
    question = body["prompt"]

    assistent = ChatAI("http://localhost:11434/api/","gemma:2b")
    response = assistent.AskQuestion(question)
    return jsonify(response), 200


if __name__ == '__main__':
    UpdateModelsList()
    UpdateConfigurationList()
    app.run(host="0.0.0.0", port=5001, debug=True)

