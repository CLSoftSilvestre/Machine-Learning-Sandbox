
from sklearn import linear_model
from sklearn import neighbors
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from PredictionModel import PredictionModel, InputFeature, ModelInformation, ReturnFeature
from utils import CreateImage
from outlierextractor import OutlierExtractor

import sys
import pandas as pd

# Models Parameters
class model_data_input():
    def __init__(self):
        df_y = "None"
        df_y_name = "None"
        df_x = "None"
        units = "None"

class base_params():
    def __init__(self):
        self.name = ""
        self.description = ""
        self.keywords = ""
        self.scaling = False
        self.featureRed = False
        self.testSize = 0.33
        self.selectKBest = 5
        self.data = model_data_input()

class knn_regressor_params(base_params):
    def __init__(self):
        self.n_neighbors=5
        self.weights='uniform'
        self.algorithm='auto'
        self.leaf_size=30
        self.p=2
        self.metric='minkowski'
        self.metric_params=None
        self.n_jobs=None

class svm_regressor_params(base_params):
    def __init__(self):
        self.kernel='rbf'
        self.degree=3
        self.gamma='scale'
        self.coef0=0.0
        self.tol=0.001
        self.C=1.0
        self.epsilon=0.1
        self.shrinking=True
        self.cache_size=200
        self.verbose=False
        self.max_iter=-1

class random_forest_regressor_params(base_params):
    def __init__(self):
        self.n_estimators=100
        self.criterion='squared_error'
        self.max_depth=None
        self.min_samples_split=2
        self.min_samples_leaf=1
        self.min_weight_fraction_leaf=0.0
        self.max_features=1.0
        self.max_leaf_nodes=None
        self.min_impurity_decrease=0.0
        self.bootstrap=True
        self.oob_score=False
        self.n_jobs=None
        self.random_state=None
        self.verbose=0
        self.warm_start=False
        self.ccp_alpha=0.0
        self.max_samples=None
        self.monotonic_cst=None

# Creates and returns one PredictionModel object with Linear regression
def LinearRegression(params : base_params = 0):

    if params.scaling:
        if params.featureRed:
            linear = make_pipeline(StandardScaler(),SelectKBest(f_classif, k=params.selectKBest), linear_model.LinearRegression())
        else:
            linear = make_pipeline(StandardScaler(), linear_model.LinearRegression())
    else:
        if params.featureRed:
            linear = make_pipeline(SelectKBest(f_classif, k=params.selectKBest), linear_model.LinearRegression())
        else:
            #linear = make_pipeline(OutlierExtractor() ,linear_model.LinearRegression())
            linear = linear_model.LinearRegression()
        
    # Set train/test groups
    x_train, x_test, y_train, y_test = train_test_split(params.data.df_x, params.data.df_y, test_size=params.testSize, random_state=42)

    # Train model
    linear.fit(x_train, y_train)
    y_pred = linear.predict(x_test)

    # Save model
    inputFeatures = []
    for item in params.data.df_x:
        inputFeatures.append(InputFeature(item, str(type(params.data.df_x[item][0])), "Description of " + item))
    
    # Calculate feature importances and update feature item.
    desc = pd.DataFrame(params.data.df_x)

    try:     
        importance = linear.coef_
        for i, v in enumerate(importance):
            inputFeatures[i].setImportance(v)
    except:
        pass

    desc = pd.DataFrame(params.data.df_x)

    for i in range(len(inputFeatures)):
            featureName = inputFeatures[i].name
            inputFeatures[i].setDescribe(desc[featureName].describe())

    pModel = PredictionModel()
    pModel.Setup(params.name, params.description, params.keywords, linear, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

    pModel.SetTestData(y_test, y_pred)

    pModel.SetTrainImage(CreateImage(y_test, y_pred, params.data.df_y_name))

    return pModel
    
# Creates and returns one PredictionModel object with KNN regression
def KnnRegression(df_y, df_y_name, df_x, units, params : knn_regressor_params = 0):

    # Check if the slectedkbestk is equal or higer thant the amount of features
    featuresCount = len(df_x.columns)
    if(params.selectKBest >= featuresCount):
        params.selectKBest = featuresCount -1

    if params.scaling:
        if params.featureRed:
            knn = make_pipeline(StandardScaler(),SelectKBest(f_classif, k=params.selectKBest), neighbors.KNeighborsRegressor(n_neighbors=params.n_neighbors, weights=params.weights, algorithm=params.algorithm, leaf_size=params.leaf_size, p=params.p, metric=params.metric, metric_params=params.metric_params, n_jobs=params.n_jobs))
        else:
            knn = make_pipeline(StandardScaler(), neighbors.KNeighborsRegressor(n_neighbors=params.n_neighbors, weights=params.weights, algorithm=params.algorithm, leaf_size=params.leaf_size, p=params.p, metric=params.metric, metric_params=params.metric_params, n_jobs=params.n_jobs))
    else:
        if params.featureRed:
            knn = make_pipeline(SelectKBest(f_classif, k=params.selectKBest), neighbors.KNeighborsRegressor(n_neighbors=params.n_neighbors, weights=params.weights, algorithm=params.algorithm, leaf_size=params.leaf_size, p=params.p, metric=params.metric, metric_params=params.metric_params, n_jobs=params.n_jobs))
        else:
            knn = neighbors.KNeighborsRegressor(n_neighbors=params.n_neighbors, weights=params.weights, algorithm=params.algorithm, leaf_size=params.leaf_size, p=params.p, metric=params.metric, metric_params=params.metric_params, n_jobs=params.n_jobs)
            #knn = make_pipeline(steps=[('Outlier extractor',OutlierExtractor()),('KNN Estimator', neighbors.KNeighborsRegressor(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf_size))])
    
    # Set train/test groups
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=params.testSize, random_state=42)

    # Train model
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    # Save model
    inputFeatures = []
    for item in df_x:
        inputFeatures.append(InputFeature(item, str(type(df_x[item][0])), "Description of " + item))
    
    # Set the feature unit
    try:
        for feature in inputFeatures:
            for funit in units:
                if feature.name == funit[0]:
                    feature.setUnit(funit[1])
    except:
        print("Error setting feature units.", file=sys.stderr)
    
    # Calculate feature importances and update feature item.
    results = permutation_importance(knn, x_train, y_train, scoring='r2')
    #print(results, file=sys.stderr)

    importance = results.importances_mean

    desc = pd.DataFrame(df_x)

    # Number of features
    n = len(inputFeatures)

    for i, v in enumerate(importance):
        if params.featureRed == False:
            inputFeatures[i].setImportance(v*n)
        featureName = inputFeatures[i].name
        inputFeatures[i].setDescribe(desc[featureName].describe())
    
    pModel = PredictionModel()
    pModel.Setup(params.name, params.description, params.keywords, knn, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))
    pModel.SetTestData(y_test, y_pred)
    pModel.SetTrainImage(CreateImage(y_test, y_pred, df_y_name))

    return pModel
