
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


import sys
import pandas as pd

# Creates and returns one PredictionModel object with Liner regression
def LinearRegression(name, description, df_y, df_y_name, df_x, scaler=False, featureReduction=False):

    if scaler:
        if featureReduction:
            linear = make_pipeline(StandardScaler(),SelectKBest(f_classif, k="all"), linear_model.LinearRegression())
        else:
            linear = make_pipeline(StandardScaler(), linear_model.LinearRegression())
    else:
        if featureReduction:
            linear = make_pipeline(SelectKBest(f_classif, k="all"), linear_model.LinearRegression())
        else:
            linear = linear_model.LinearRegression()
        
    # Set train/test groups
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)

    # Train model
    linear.fit(x_train, y_train)
    y_pred = linear.predict(x_test)

    # Save model
    inputFeatures = []
    for item in df_x:
        inputFeatures.append(InputFeature(item, str(type(df_x[item][0])), "Description of " + item))
    
    # Calculate feature importances and update feature item.
    desc = pd.DataFrame(df_x)

    try:     
        importance = linear.coef_
        for i, v in enumerate(importance):
            inputFeatures[i].setImportance(v)
    except:
        pass

    desc = pd.DataFrame(df_x)

    for i in range(len(inputFeatures)):
            featureName = inputFeatures[i].name
            inputFeatures[i].setDescribe(desc[featureName].describe())

    pModel = PredictionModel()
    pModel.Setup(name,description,linear, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

    pModel.SetTrainImage(CreateImage(y_test, y_pred, df_y_name))

    return pModel
    
# Creates and returns one PredictionModel object with KNN regression
def KnnRegression(name, description, df_y, df_y_name, df_x, units, scaler=False, featureReduction=False, n=5, weights='distance', algorithm='auto', leaf_size=30):

    if scaler:
        if featureReduction:
            knn = make_pipeline(StandardScaler(),SelectKBest(f_classif, k="all"), neighbors.KNeighborsRegressor(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf_size))
        else:
            knn = make_pipeline(StandardScaler(), neighbors.KNeighborsRegressor(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf_size))
    else:
        if featureReduction:
            knn = make_pipeline(SelectKBest(f_classif, k="all"), neighbors.KNeighborsRegressor(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf_size))
        else:
            knn = neighbors.KNeighborsRegressor(n_neighbors=n, weights=weights, algorithm=algorithm, leaf_size=leaf_size)
    
    # Set train/test groups
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)

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
    importance = results.importances_mean

    desc = pd.DataFrame(df_x)

    for i, v in enumerate(importance):
        inputFeatures[i].setImportance(v)
        featureName = inputFeatures[i].name
        inputFeatures[i].setDescribe(desc[featureName].describe())
    
    pModel = PredictionModel()
    pModel.Setup(name,description,knn, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

    pModel.SetTrainImage(CreateImage(y_test, y_pred, df_y_name))

    return pModel
