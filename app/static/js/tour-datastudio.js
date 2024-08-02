
// Check if the user as already run the tour
let x = getCookie("tourDataStudioPart1Complete")
let y = getCookie("tourDataStudioPart2Complete")

var nodatasetimported

var intro = introJs();

try {
    nodatasetimported = document.getElementById("notuploaded").innerText
} catch (error) {
    nodatasetimported = ""
}

importView = false

if (nodatasetimported == "Load dataset"){
    importView = true
}

// Tour part 1 - Importing file
if ( (x== "" || x == "false") && importView){
    introJs().setOptions({
        showProgress: true,
        steps: [{
            title: "DataStudio - Importing dataset",
            intro: "This tutorial will show you how to import your dataset to the platform."
        },
        {
            element: document.getElementById('tour-train-import-file'),
            title: "Importing dataset file",
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
            position: 'left'
        }]
    }).start().oncomplete(function () {
        setCookie("tourDataStudioPart1Complete", true, 90);
    });
}

// Tour part 2 - View Data
if ( (y== "" || y == "false") && !importView){
    intro.setOptions({
        showProgress: true,
        exitOnOverlayClick: false,
        steps: [{
            title: "Analyze your data",
            intro: "This tutorial you will check how to analyse, clean and prepare your data before consuming it in model training."
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
            intro: 'This is a very important step on the data preparation. Please ensure that your data dont contain null values. Click here to delete rows containing null values. In some cases its prefered to use this command after filtering the data.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-script'),
            title: "Python scripts",
            intro: 'Here you can perform some basic Python scripts to modify the dataset. For instance add calculated columns or perform math operations.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-correlation'),
            title: "Correlation map",
            intro: 'Here you can can access the correlation map between the variable.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-scatter'),
            title: "Scatter plots",
            intro: 'This is a very usefull feature to check the correlation of your variables. Here you can correlate 2 variables in X,Y axis and use a third one for the color. Automatic linear and polymonial regression are created.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-export'),
            title: "Download dataset",
            intro: 'This allows the exportation of the current state of the dataset in CSV format.',
            position: 'left'
        },
        {
            element: document.getElementById('tour-train-import-useml'),
            title: "Use dataset for training ML",
            intro: 'Use the actual cleaned dataset for training machine learning algorithm. Please remove all columns not required before performing this operation.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-variables'),
            title: "Dataset variables",
            intro: 'Here you can check all the columns detected in the CSV dataset.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-datatypes'),
            title: "Variables datatypes",
            intro: 'Here you can check the actual variable datatype. OBJECTS datatypes cannot be used for ML training. Please remove it before.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-settype'),
            title: "Set column datatype.",
            intro: 'Use this command if you want to change the variable type.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-remove'),
            title: "Remove columns",
            intro: 'Select the columns that will be removed from the dataset. The columns are removed from the memory dataset not from the CSV file.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-filter'),
            title: "Filter column data",
            intro: 'Filter the rows according the interval of data defined.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-setname'),
            title: "Set column name.",
            intro: 'Use this command if you want to change the variable name. For better results dont use spaces or special characters.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-desc'),
            title: "Data Statistics",
            intro: 'This tab contains the statistics details of the data.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-outliers'),
            title: "Boxplots - Outliers",
            intro: 'This tab contains the boxplots of all the variables and the outliers identified as red circles. If chart is not showed probably the data is not compatible (strings or nulls).',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-import-operations'),
            title: "Data operations",
            intro: 'In this tab user can view the list of operations performed and delete if required.',
            position: 'left'
        },
        {
            element: document.getElementById('tour-train-import-dataview'),
            title: "Data preview",
            intro: 'Here users can preview the first 10 records of the actual dataset.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-datastudio-console'),
            title: "Console",
            intro: 'Here users can access the console output.',
            position: 'left'
        }]
    }).start().oncomplete(function () {
        setCookie("tourDataStudioPart2Complete", true, 90);
    });
}

// Activate tabs and modals allong the tutorial
intro.onchange(function(element) {

    if(element.id === 'tour-train-import-script'){
        $("#script").modal("show");
    } else {
        $("#script").modal("hide");
    }

    if(element.id === 'tour-train-correlation'){
        $("#correlation").modal("show");
    } else {
        $("#correlation").modal("hide");
    }

    if(element.id === 'tour-train-scatter'){
        $("#scatterplot").modal("show");
    } else {
        $("#scatterplot").modal("hide");
    }

    if(element.id === 'tour-train-import-desc'){
        document.getElementById("descri-tab").click()
    }

    if(element.id === 'tour-train-import-outliers'){
        document.getElementById("outliers-tab").click()
    }

    if(element.id === 'tour-train-import-operations'){
        document.getElementById("dataoperations-tab").click()
    }

});
