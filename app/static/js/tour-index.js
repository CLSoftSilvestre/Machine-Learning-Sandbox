
// Check if the user as already run the tour
let x = getCookie("tourIndexComplete")

if ( (x== "" || x == "false")){
    introJs().setOptions({
        showProgress: true,
        steps: [{
            title: "Main page tutorial",
            intro: "Please take a few moment to have a quick tour on the main features of this platform. Here you can take a look how to easily import, prepare, model and consume your data. During this process you will also learn the key concepts of the process."
        },
        {
            element: document.getElementById('tour-index-datastudio'),
            title: "Data Studio",
            intro: 'Here is where the process begins. In this area you can import your dataset, analyse and prepare your data before training models.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-index-importmodel'),
            title: "Model importation",
            intro: 'If you already have one model created in other instance (server) you can easily import to this platfom.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-index-refresh'),
            title: "Refresh model list",
            intro: 'Refresh the models list available in this server.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-index-modelslist'),
            title: "Available models",
            intro: 'Here you can check the list of available trained models in the server.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-index-rmse'),
            title: "Model accuracy - RMSE",
            intro: 'RMSE is commonly used to compare the performance of different regression models and to assess the quality of the models predictions. It measures the average magnitude of the error between the predicted values and the actual values in a regression task. Lower RMSE values indicate that the models predictions are closer to the actual values, while higher RMSE values indicate larger prediction errors',
            position: 'left'
        },
        {
            element: document.getElementById('tour-index-r2'),
            title: "Model accuracy - R^2",
            intro: 'R-squared is a valuable metric for understanding how well a regression model captures the variation in the dependent variable and is widely used in regression analysis for evaluating model performance. It provides insight into the goodness of fit of the model, indicating how well the independent variables explain the variability of the dependent variable.',
            position: 'left'
        },
        {
            element: document.getElementById('tour-index-commands'),
            title: "Model commands",
            intro: 'In this group of buttons you can access the available command to each model.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-index-predict'),
            title: "Use prediction",
            intro: 'Use this button to create a single prediction. Here you will be able to input the data required to calculate one prediction using the selected model.',
            position: 'left'
        },
        {
            element: document.getElementById('tour-index-batch-predict'),
            title: "Use batch prediction",
            intro: 'Use this button to create a batch prediction. Here you will be able to input one CSV file with the features data to create several prediction at once.',
            position: 'left'
        },
        {
            element: document.getElementById('tour-index-info'),
            title: "Model information",
            intro: 'Check here the model information like algorithm used, correlations matrix, training charts, etc...',
            position: 'left'
        },
        {
            element: document.getElementById('tour-index-download'),
            title: "Download model",
            intro: 'Here you can download the trained model and use it in python applications. Source code required is included!',
            position: 'left'
        },
        {
            element: document.getElementById('tour-index-delete'),
            title: "Delete model",
            intro: 'This button deletes the model from the server. Dont worry confirmation is required!',
            position: 'left'
        },
        {
            title: "Hide / show tutorials",
            intro: 'To show / hide the learning tutorials check the selector in the About section.',
            position: 'left'
        }]
    }).start().oncomplete(function () {
        setCookie("tourIndexComplete", true, 90);
    });
}
