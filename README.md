# MLS - Machine learning sandox
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
### Classification models
- Random forest
- K Nearest Neighbors
***
## Printscreens of the app

### Main screen
Management of existent models
![alt text](./screenshots/mainscreen.png?raw=true)

### Train  models screen
Train model with imported csv dataset. Some features implemented to clean the data, delete columns, select the Y feature. The system show the main stats of the data imported and creates a correlation matrix.
![alt text](./screenshots/trainmodel.png?raw=true)

### Check the model performance
After training the model, it will be saved in the folder ./app/models/. The user can check the performance of the model accesing the model details in the main screen of the application. It will be visible the chart with the Real vs Predicted values (limited to 100 tests) created during the training and will be shown the R^2^ and MSE scores.
![alt text](./screenshots/predictions.jpg?raw=true)

### Download trained model
Use the trained model in other python applications.
![alt text](./screenshots/download.png?raw=true)

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
