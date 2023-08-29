# MLS - Machine learning sandox
Web application to train and use machine learning models with integrated API.

## Printscreens of the app

### Main screen
Management of existent models
![alt text](https://github.com/CLSoftSilvestre/FactoryFloor/blob/main/screenshots/mainscreen.png?raw=true)

### Train  models screen
Train model with imported csv dataset. Some features implemented to clean the data, delete columns, select the Y feature. The system show the main stats of the data imported and creates a correlation matrix.
![alt text](https://github.com/CLSoftSilvestre/FactoryFloor/blob/main/screenshots/trainmodel.png?raw=true)


## How to run

### Activate the environment
https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/
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
