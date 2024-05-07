
// Check if the user as already run the tour
let y = getCookie("tourTrainComplete")

// Tour part 2 - View Data
if ( (y== "" || y == "false") ){
    introJs().setOptions({
        showProgress: true,
        steps: [{
            title: "Prepare data for training",
            intro: "This tutorial you will check how to provide the final information and select the model"
        },
        {
            element: document.getElementById('tour-train-datastudio'),
            title: "Back to Data Studio",
            intro: 'Return to data studio to perform modifications to dataset.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-setunits'),
            title: "Set units",
            intro: 'Select the units of each variable.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-sety'),
            title: "Set dependent variable",
            intro: 'Select the variable that you want to be predicted.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-train-selectmodel'),
            title: "Select the model",
            intro: 'Select the model that best suits your needs.',
            position: 'right'
        },
        {
            title: "Train time",
            intro: 'Now that you have your dataset ready, its time to train your model. Please select the kind of model that you want to train.',
        }]
    }).start().oncomplete(function () {
        setCookie("tourTrainComplete", true, 90);
    });
}
