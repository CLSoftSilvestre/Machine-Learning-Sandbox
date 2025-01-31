# MLS - Machine learning sandbox
Flask Web application to train and use machine learning models with integrated API and testing / simulation environment. MLS can connect directly to data on the field using several industrial protocols like Siemens S7 connection, MQTT, OPC-UA...
***
## Available ML models
The platform can train several types of models including classification and regressions.
Standard scaler and feature reduction can also be applied in the pipeline by selecting the desired functionalities in the training process.

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

### Anomaly detection models (under development)
- Local Outlier Factor (LOF)
- Isolation forest


## Application showcase

### Main screen
On the main screen user can check the available trained models, quickly check the status of the automation flow and access to the main actions (Single prediction, Batch prediction, Information, Automation Flow, Download and Delete)
![alt text](./screenshots/mainscreen.jpeg?raw=true)

### Data Studio
In the Data Studio the data can be imported using several connectors. (more to be added in the future).
![alt text](./screenshots/Datastudio_import.jpeg?raw=true)

From this module, data can be cleaned and analysed before imported into the model training area
![alt text](./screenshots/datastudio.png?raw=true)

![alt text](./screenshots/boxplots.png?raw=true)

Several operation can be performed to the dataset and if needed data transformation steps can be changed or deleted.

![alt text](./screenshots/dataoperations.png?raw=true)

It's also possible to add python code to perform operation in the Dataset
![alt text](./screenshots/scripting.png?raw=true)

And check the correlations and show a scatter plot between the available variables.
![alt text](./screenshots/correlations.png?raw=true)
![alt text](./screenshots/scatterplot.png?raw=true)

### Train  models screen
After cleaned and analysed, the dataset can be imported to the model training area.
Here the user can set the details of the model and select the model algorithm.
![alt text](./screenshots/models.jpeg?raw=true)

### Check the model performance
After training the model, it will be saved in the folder ./app/models/. The user can check the performance of the model accesing the model details in the main screen of the application. It will be visible the chart with the Real vs Predicted values (limited to 100 tests) created during the training and will be shown the R^2^ and MSE scores.
![alt text](./screenshots/details.jpeg?raw=true)

### Model automated simulation flow
Once created the models, the user can connect the model to sources of data or directly to equipments to analyse the model with live data. Users can use this enviroment to simulate several working conditions and output the values to CSV file to be analysed. (This feature is currently only available for regression models)
Several connector are available as:
- OSIsoft PI (using PISDK only available for windows)
- MQTT
- Siemens S7 connector
- OPC UA (under development)
- Bluetooth Low Energy
- Modbus TCP
- ...

![alt text](./screenshots/automationflow.png?raw=true)

### Using API
The integrated API prodived methods to list the available models, get the required input fields and retrieve the prediction.

#### Getting the list of available trained models
The get method to the address <http://{serveraddress}/api/GetModels> return the list of available models in the server. The uuid of the model will be required for the other endpoints.

![alt text](./screenshots/apigetmodels.jpg?raw=true)

#### Getting the list inputs for the model
The get method to the address <http://{serveraddress}/api/Predict/{model_uuid}> return the list of input parameters of the model.

![alt text](./screenshots/apipredictget.jpg?raw=true)

#### Predicting the value
The post method to the address <http://{serveraddress}/api/Predict/{model_uuid}> return the information regarding the mode, the list of input parameters and the prediction result.

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
- The folder `./app/models/` will be shared with the docker container to store the models created.
- The folder `./app/config/` will be shared with the docker container to store the app configuration file.
