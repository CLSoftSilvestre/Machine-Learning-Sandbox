# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:53:44 2023

@author: CSilvestre
"""

from flask import Flask, render_template, send_from_directory, request, redirect, url_for, jsonify, session
from flask_session import Session

from ModelManager import ModelManager
import pandas as pd
import sys
import matplotlib.pylab as plt
import seaborn as sn
import numpy as np

from sklearn import neighbors
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif

from models import LinearRegression


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from PredictionModel import PredictionModel, InputFeature, ModelInformation, ReturnFeature
from ModelManager import ModelManager
from Database import UserLogin, User

import os
import io
import base64
import json
from werkzeug.utils import secure_filename


app = Flask(__name__, instance_relative_config=True)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['SECRET'] = "secret!123"
Session(app)

mm = ModelManager()
modelsList = []
appversion = "1.2.7"
model_version = 4

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
        admin = session.get('autenticated')
        name = session.get('user')
    else:
        admin = False
        name = ""

    template_config = {'admin': admin, 'username': name}

    return template_config

@app.route("/index")
@app.route("/")
def index():

    # Load the models in the first usage
    if len(modelsList) == 0:
        UpdateModelsList()

    return render_template('index.html', models=modelsList)

@app.route("/details/<uuid>", methods=['GET'])
def details(uuid):
    for model in modelsList:
        if (model.uuid == uuid):
            #print(model.name, file=sys.stderr)
            try:
                image = model.imageData
            except:
                image = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="

            try:
                image2 = model.correlationMatrixImage
            except:
                image2 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="

            return render_template('details.html', Model=model, imageData=image, correlationImageData=image2)
  
    return render_template('details.html')

@app.route("/download/<uuid>", methods=['GET'])
def download(uuid):

    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    for model in modelsList:
        if (model.uuid == uuid):
            return render_template('download.html', Model=model)
        
    return render_template('download.html')

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
            # Calculate the min and max expected according R2 score
            minResult = resultValue - (resultValue * ((1-activeModel.R2)/2))
            maxResult = resultValue + (resultValue * ((1-activeModel.R2)/2))

            return render_template('predict.html', Model=activeModel, Result="{:,}".format(round(resultValue,2)), Features=features, MinResult="{:,}".format(round(minResult,2)), MaxResult="{:,}".format(round(maxResult,2))) 
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
        print(request.form['mod'], file=sys.stderr)
        print(type(request.form['mod']), file=sys.stderr)
        
        if (request.form['mod']=="clearnull"):     
            session['temp_df'] = session['temp_df'].dropna(how='any', axis=0)
            session['temp_df'] = session['temp_df'].reset_index(drop=True)

        elif (request.form['mod']=="remcol"):
            session['temp_df'] = session['temp_df'].drop([request.form['column']], axis=1)

            if(session['temp_df_y_name'] != ""):
                session['temp_df_x'] = session['temp_df_x'].loc[:,session['temp_df'].columns != session['temp_df_y_name']]

        elif(request.form['mod']=="setdependent"):
            session['temp_df_y'] = session['temp_df'][request.form['column']]
            session['temp_df_y_name'] = request.form['column']
            session['temp_df_x'] = session['temp_df'].loc[:,session['temp_df'].columns != session['temp_df_y_name']]

        elif(request.form['mod']=="setproperties"):
            propList = []
            session['temp_df_units'] = []

            for i in session['temp_df']:
                varUnit = request.form[i]
                print("Objecto " + i + " - Unidade: " + varUnit, file=sys.stderr)
                propList.append((i, varUnit))
                session['temp_df_units'] = propList


        # Update the correlation matrix image
        session['temp_df'].corr(method="pearson")
        corr_matrix = session['temp_df'].corr(min_periods=1)
        sn.heatmap(corr_matrix, cbar=0, annot=True, fmt=".1f", linewidths=2,vmax=1, vmin=0, square=True, cmap='Greens')

        # Save image data in variable
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg', bbox_inches='tight', pad_inches=0.0)
        my_stringIObytes.seek(0)
        session['heatmap_base64_jpgData'] = base64.b64encode(my_stringIObytes.read()).decode()
        plt.clf()

    if session['temp_df'].columns.size > 0:   
        return render_template('train.html', tables=[session['temp_df'].head(n=10).to_html(classes='table table-hover table-sm text-center table-bordered', header="true")], titles=session['temp_df'].columns.values, uploaded=True, descTable=[session['temp_df'].describe().to_html(classes='table table-hover text-center table-bordered', header="true")], datatypes = session['temp_df'].dtypes, dependend = session['temp_df_y_name'], heatmap=session['heatmap_base64_jpgData'])
    else:
        return render_template('train.html')

@app.route("/uploader/", methods=['GET', 'POST'])
def uploader():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')

    if request.method == 'POST':
        f = request.files['file']
        if f:
            # Process the file
            sep = request.form['sep']
            dec = request.form['dec']

            session['temp_df'] = pd.read_csv(f,sep=sep, decimal=dec)

            # Update the correlation matrix image
            session['temp_df'].corr(method="pearson")
            corr_matrix = session['temp_df'].corr(min_periods=1)
            sn.heatmap(corr_matrix, cbar=0, annot=True, fmt=".1f", linewidths=2,vmax=1, vmin=0, square=True, cmap='Greens')

            # Save image data in variable
            my_stringIObytes = io.BytesIO()
            plt.savefig(my_stringIObytes, format='jpg', bbox_inches='tight', pad_inches=0.0)
            my_stringIObytes.seek(0)
            session['heatmap_base64_jpgData'] = base64.b64encode(my_stringIObytes.read()).decode()
            plt.clf()

            return redirect('/train')
        else:
            return redirect('/train')
    else:
        return redirect('/train')

@app.route("/cleardataset/", methods=['GET'])
def cleardataset():
    session['temp_df'] = pd.DataFrame()
    session['temp_df_y'] = pd.DataFrame()
    session['temp_df_y_name'] = ""
    session['temp_df_x'] = pd.DataFrame()
    session['heatmap_base64_jpgData'] = ""

    return render_template('train.html')

# Start of Models training
@app.route("/linear/", methods=['GET', 'POST'])
def linear():
    if (session.get('autenticated') != True):
        return redirect('/notauthorized')
    
    if(request.method == 'POST'):

        # Set the model
        name = request.form['name']
        description = request.form['description']
        scaling = bool(request.form.get('scaling'))
        featurered = bool(request.form.get('featurered'))

        pModel = LinearRegression(name, description, session['temp_df_y'],session['temp_df_y_name'], session['temp_df_x'], scaling, featurered)
        pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)

        mMan = ModelManager()
        modelFileName = name + ".model"
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
        return render_template('linear.html')

@app.route("/knnreg/", methods=['GET', 'POST'])
def knnreg():
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

        if scaling:
            if featurered:
                knn = make_pipeline(StandardScaler(),SelectKBest(f_classif, k="all"), neighbors.KNeighborsRegressor(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf))
            else:
                knn = make_pipeline(StandardScaler(), neighbors.KNeighborsRegressor(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf))
        else:
            if featurered:
                knn = make_pipeline(SelectKBest(f_classif, k="all"), neighbors.KNeighborsRegressor(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf))
            else:
                knn = neighbors.KNeighborsRegressor(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf)
        
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

        # Calculate feature importances and update feature item.
        results = permutation_importance(knn, x_train, y_train, scoring='r2')
        importance = results.importances_mean
        desc = pd.DataFrame(session['temp_df_x'])

        for i, v in enumerate(importance):
            inputFeatures[i].setImportance(v)
            featureName = inputFeatures[i].name
            inputFeatures[i].setDescribe(desc[featureName].describe())

        pModel = PredictionModel()
        pModel.Setup(name,description,knn, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

        pModel.SetTrainImage(CreateImage(y_test, y_pred))
        pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)

        mMan = ModelManager()
        modelFileName = name + ".model"
        filepath = os.path.join(app.root_path, 'models', modelFileName)

        print("filepath: ", filepath, file=sys.stderr)

        mMan.SaveModel(pModel, filepath)

        UpdateModelsList()

        return redirect('/index')
    
    else:
        return render_template('knnreg.html')

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
        
        # Calculate feature importances and update feature item.
        for i in enumerate(inputFeatures):
            inputFeatures[i].setImportance(0)
        
        # Calculate feature importances and update feature item.
        results = permutation_importance(knn, x_train, y_train, scoring='accuracy')
        importance = results.importances_mean
        desc = pd.DataFrame(session['temp_df_x'])

        for i, v in enumerate(importance):
            inputFeatures[i].setImportance(v)
            featureName = inputFeatures[i].name
            inputFeatures[i].setDescribe(desc[featureName].describe())

        pModel = PredictionModel()
        pModel.Setup(name,description,knn, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

        pModel.SetTrainImage(CreateImage(y_test, y_pred))
        pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)

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
        except:
            print("Error setting feature units.", file=sys.stderr)

        
        # Calculate feature importances and update feature item.
        importance = clf.feature_importances_
        desc = pd.DataFrame(session['temp_df_x'])

        for i, v in enumerate(importance):
            inputFeatures[i].setImportance(v)
            featureName = inputFeatures[i].name
            inputFeatures[i].setDescribe(desc[featureName].describe())

        pModel = PredictionModel()
        pModel.Setup(name,description,clf, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

        pModel.SetTrainImage(CreateImage(y_test, y_pred))
        pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)

        mMan = ModelManager()
        modelFileName = name + ".model"
        filepath = os.path.join(app.root_path, 'models', modelFileName)

        print("filepath: ", filepath, file=sys.stderr)

        mMan.SaveModel(pModel, filepath)

        UpdateModelsList()

        return redirect('/index')
    
    else:
        return render_template('randomforest.html')

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
        pModel.Setup(name,description,clf, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

        pModel.SetTrainImage(CreateImage(y_test, y_pred))
        pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)

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

        if scaling:
            if featurered:
                clf = make_pipeline(StandardScaler(), SelectKBest(f_classif, k="all"), tree.DecisionTreeRegressor(max_depth=max_depth, criterion=criterion))
            else:
                clf = make_pipeline(StandardScaler(), tree.DecisionTreeRegressor(max_depth=max_depth, criterion=criterion))
        else:
            if featurered:
                clf = make_pipeline(SelectKBest(f_classif, k="all"), tree.DecisionTreeRegressor(max_depth=max_depth, criterion=criterion))
            else:
                clf = tree.DecisionTreeRegressor(max_depth=max_depth, criterion=criterion)

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
        pModel.Setup(name,description,clf, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

        pModel.SetTrainImage(CreateImage(y_test, y_pred))
        pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)

        mMan = ModelManager()
        modelFileName = name + ".model"
        filepath = os.path.join(app.root_path, 'models', modelFileName)

        print("filepath: ", filepath, file=sys.stderr)

        mMan.SaveModel(pModel, filepath)

        UpdateModelsList()

        return redirect('/index')
    
    else:
        return render_template('treereg.html')

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
        pModel.Setup(name,description,clf, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

        pModel.SetTrainImage(CreateImage(y_test, y_pred))
        pModel.SetCorrelationMatrixImage(session['heatmap_base64_jpgData'])
        pModel.SetModelVersion(model_version, appversion)

        mMan = ModelManager()
        modelFileName = name + ".model"
        filepath = os.path.join(app.root_path, 'models', modelFileName)

        print("filepath: ", filepath, file=sys.stderr)

        mMan.SaveModel(pModel, filepath)

        UpdateModelsList()

        return redirect('/index')

    else:
        return render_template('perceptronreg.html')

# End of Models training

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
    #modelspath = os.path.join(app.instance_path, 'models', "*.model")
    modelsList = mm.GetModelsList(modelspath)
    print(app.instance_path, file=sys.stderr)

def CreateImage(test, pred):
    # Add train image into model
    plot_y_test = test.reset_index()
    del plot_y_test['index']

    plt.plot(plot_y_test[0:100], color='#2c3e50', label='Real')
    plt.plot(pred[0:100], color='#18bc9c', label='Predicted')
    plt.xlabel('Predictions')
    plt.ylabel(session['temp_df_y_name'])
    plt.legend(loc='lower right')

    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
    plt.clf()

    return my_base64_jpgData

@app.route("/login/", methods=['POST'])
def Login():

    name = request.form['username']
    password = request.form['pswd']

    params = (name, password)
    dbPath = os.path.join(app.root_path, 'database', "mls.db")
    user = UserLogin(dbPath, params)

    if (user != False):
        session['user'] = user.name
        session['role'] = user.role
        session['autenticated'] = True

        # Train temp variables
        # Use session variables to store temp data
        # This will avoid setting data for other users
        session['temp_df'] = pd.DataFrame()
        session['temp_df_y'] = pd.DataFrame()
        session['temp_df_y_name'] = ""
        session['temp_df_x'] = pd.DataFrame()
        session['heatmap_base64_jpgData'] = ""
        session['temp_df_units'] = []
    else:
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
    session['temp_df_units'] = []

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

        for key in request.form:
            if (len(request.form[key])==0):
                # Is empty and should be number
                return redirect(url_for('index'))
            
            inputData[key]=[request.form[key]]
            features.append(ReturnFeature(key, float(inputData[key])))
            #featuresJson = json.dumps([obj.__dict__ for obj in features])
            featuresJson = json.dumps(features, default=ReturnFeature.serialize)

        # Check the model
        for model in modelsList:
            if (model.uuid == uuid):
                #activeModel = model
                try:
                    result =model.model.predict(inputData)
                    
                    try:
                        innerResult = result[0][0]
                    except:
                        innerResult = result[0]
                    
                    minResult = innerResult - (innerResult * ((1-model.R2)/2))
                    maxResult = innerResult + (innerResult * ((1-model.R2)/2))

                    data = {
                        "UUID" : model.uuid,
                        "Model" : model.name,
                        "Description" : model.description,
                        "Prediction" : innerResult,
                        "MinPrediction" : minResult,
                        "MaxPrediction" : maxResult,
                        "Features": json.loads(featuresJson)
                    }

                    return jsonify(data)
                except:
                    return "Error predicting value.", 404

if __name__ == '__main__':
    UpdateModelsList()
    app.run(host="0.0.0.0", port=80, debug=True)

