# MLS - Machine learning sandbox
Flask Web application to train and use machine learning models with integrated API.
***
## Available ML models
The platform can train several types of models including classification and regressions.
Sdandard scaler and feature reduction can also be applied in the pipeline by selecting the desired functionalities in the training process.

### Regression models
- Linear regression
- K Nearest Neighbors regressor
- Suport Vector Machine regression
- Decision tree regressor
- Random Forest regressor
- Neural Network Multi-Layer Perceptron regressor

### Classification models
- Random forest
- K Nearest Neighbors


## Printscreens of the app

### Main screen
Management of existent models
![alt text](./screenshots/index.jpg?raw=true)

### Data Studio screen
In the Data Studio the data can be imported, cleaned and analysed before imported into the model training area
![alt text](./screenshots/datastudio.jpg?raw=true)

It's also possible to add python code to perform operation in the Dataset
![alt text](./screenshots/scripting.jpg?raw=true)

And check the correlations and show a scatter plot between the available variables.
![alt text](./screenshots/scatter.jpeg?raw=true)

### Train  models screen
After cleaned and analysed, the dataset can be imported to the model training area.
Here the user can set the deatils of the model and select the model algorithm.
![alt text](./screenshots/models.jpeg?raw=true)

### Check the model performance
After training the model, it will be saved in the folder ./app/models/. The user can check the performance of the model accesing the model details in the main screen of the application. It will be visible the chart with the Real vs Predicted values (limited to 100 tests) created during the training and will be shown the R^2^ and MSE scores.
![alt text](./screenshots/details.jpeg?raw=true)

### Using API
The integrated API prodived methods to list the available models, get the required input fields and retrieve the prediction.

#### Getting the list of available trained models
The get method to the address <http://{serveraddress}/GetModels> return the list of available models in the server. The uuid of the model will be required for the other endpoints.

![alt text](./screenshots/apigetmodels.jpg?raw=true)

#### Getting the list inputs for the model
The get method to the address <http://{serveraddress}/Predict/{model_uuid}> return the list of input parameters of the model.

![alt text](./screenshots/apipredictget.jpg?raw=true)

#### Predicting the value
The post method to the address <http://{serveraddress}/Predict/{model_uuid}> return the information regarding the mode, the list of input parameters and the prediction result.

![alt text](./screenshots/apipredictpost.jpg?raw=true)

***
## How to run

### Create the environment (if not available)
```bash
python<version> -m venv <virtual-environment-name>
```

### Activate the environment
<https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/>
```bash
source env/bin/activate
```

### Instal the required libraries
```bash
pip install -r requirements.txt
```

### Run flask app
```bash
flask run
```
or
```bash
python<version> app.py
```

### Run on Docker
```bash
docker-compose up
```
The folder ./app/models/ will be shared with the docker container. Any models created in the docker environment will be saved in the models folder of the host.
