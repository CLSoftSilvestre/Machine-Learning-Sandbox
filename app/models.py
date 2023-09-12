
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from PredictionModel import PredictionModel, InputFeature, ModelInformation, ReturnFeature
from utils import CreateImage


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

    pModel = PredictionModel()
    pModel.Setup(name,description,linear, inputFeatures, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred))

    pModel.SetTrainImage(CreateImage(y_test, y_pred, df_y_name))

    return pModel
    
    