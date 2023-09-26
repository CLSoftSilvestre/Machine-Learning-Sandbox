
// Check if the user as already run the tour
let x = getCookie("tourIndexComplete")

if ( (x== "" || x == "false")){
    introJs().setOptions({
        showProgress: true,
        steps: [{
            title: "Main page tutorial",
            intro: "Please take a few moment to have a quick tour on the main features of this platform."
        },
        {
            element: document.getElementById('tour-index-newmodel'),
            title: "Create new ML model",
            intro: 'Here you can import your training data and select the algorithm to train your new model.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-index-modelslist'),
            title: "Available models",
            intro: 'Here you can check the list of available trained models in the server.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-index-r2'),
            title: "Model accuracy",
            intro: 'The model accuracy can be checked in the R2 and MSE fields. For better results use models with higher R2 and lower MSE (mean squared error).',
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
