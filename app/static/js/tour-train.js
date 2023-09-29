
// Check if the user as already run the tour
let x = getCookie("tourTrainPart1Complete")
let y = getCookie("tourTrainPart2Complete")

var nodatasetimported

try {
    nodatasetimported = document.getElementById("notuploaded").innerText
} catch (error) {
    nodatasetimported = ""
}

importView = false

if (nodatasetimported == "No dataset imported."){
    importView = true
}

// Tour part 1 - Importing file
if ( (x== "" || x == "false") && importView){
    introJs().setOptions({
        showProgress: true,
        steps: [{
            title: "Importing dataset tutorial",
            intro: "This tutorial will show you how to import your dataset to train your model."
        },
        {
            element: document.getElementById('tour-train-import-file'),
            title: "Importing dataset",
            intro: 'Select the dataset file in your PC. Please be aware that you can only import CSV files.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-separator'),
            title: "Separator",
            intro: 'Please define the type of separator used by your CSV file (tipically semi-colon or colon).',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-decimal'),
            title: "Decimal",
            intro: 'Please define the decimal point used in your CSV dataset. If your float number are set as objects, probably it is because of the decimal point.',
            position: 'left'
        },
        {
            element: document.getElementById('tour-train-import-parse'),
            title: "Parse CSV",
            intro: 'Parse the CSV file and open the data visualization screen.',
            position: 'right'
        }]
    }).start().oncomplete(function () {
        setCookie("tourTrainPart1Complete", true, 90);
    });
}

// Tour part 2 - View Data
if ( (y== "" || y == "false") && !importView){
    introJs().setOptions({
        showProgress: true,
        steps: [{
            title: "Analyze your data",
            intro: "This tutorial you will check how to check your data and select the features to train your model."
        },
        {
            element: document.getElementById('tour-train-import-delete-dataset'),
            title: "Delete dataset",
            intro: 'Using this button you can delete the dataset from the memory and import a new file.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-clear-nulls'),
            title: "Clear nulls",
            intro: 'This is a very important step on the data preparation. Please ensure that your data dont contain null values. Click here to delete rows containing null values.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-regressions'),
            title: "Regressions algorithms",
            intro: 'After preparing you data select here the regression algorithm that you want to train. Regression is used for predicting continuous values (floats).',
            position: 'left'
        },
        {
            element: document.getElementById('tour-train-import-classifications'),
            title: "Classification algorithms",
            intro: 'After preparing you data select here the classification algorithm that you want to train. Classification is used for predicting states (integer).',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-variables'),
            title: "Dataset variables",
            intro: 'Here you can check all the collumns detected in the CSV dataset. If required the headers name must be changed in the CSV and the file re-imported.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-datatypes'),
            title: "Variables datatypes",
            intro: 'The type of variable is shown in this row. Please ensure that you only use int64 or float64 variables types. Columns with object datatype should be removed.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-remove'),
            title: "Remove columns",
            intro: 'Select the columns that will be removed from the dataset. The columns are removed from the memory dataset not from the CSV file.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-sety'),
            title: "Set Y",
            intro: 'Select the column that contains the values to be predicted.',
            position: 'right'
        },
        {
            title: "Train time",
            intro: 'Now that you have your dataset ready, its time to train your model. Please select the kind of model that you want to train.',
        }]
    }).start().oncomplete(function () {
        setCookie("tourTrainPart2Complete", true, 90);
    });
}
