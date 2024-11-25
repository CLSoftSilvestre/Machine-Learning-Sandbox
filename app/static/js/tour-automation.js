// Check if the user as already run the tour
let y = getCookie("tourAutomationComplete")

// Tour part 2 - View Data
if ( (y== "" || y == "false") ){
    introJs().setOptions({
        showProgress: true,
        steps: [{
            title: "Automatize the testing of the model",
            intro: "This tutorial you will check how to automatize the model and connect to several industrial protocol. The goal of the flow is to test scenarios and output values to charts or CSV files. This module is not intent to create permanent connections to equipments."
        },
        {
            element: document.getElementById('tour-automation-nodes'),
            title: "Available nodes",
            intro: 'Here, the use can access to the available list of node to use in the automation flow. To use it, just drag and drop to the flow canvas.',
            position: 'right'
        },
        {
            element: document.getElementById('tour-automation-canvas'),
            title: "Connecting nodes",
            intro: 'To connect nodes, darg the output of one node until the input of another. Some node are not compatible eachother. If a bad connection is detected, the user will be informed.',
            position: 'right'
        },
        {
            element: document.getElementById('btnSaveFlow'),
            title: "Deploy flow",
            intro: 'Before using the flow, user need to deploy to the server. (Its recomened to Deploy even no modifications were performed)',
            position: 'right'
        },
        {
            element: document.getElementById('btnStartFlow'),
            title: "Start flow",
            intro: 'Here user can start the flow and follow the calculated outputs.',
            position: 'right'
        },
        {
            element: document.getElementById('flow-status-text'),
            title: "Flow status",
            intro: 'Here user can check the status of the flow.',
            position: 'right'
        },
        {
            element: document.getElementById('btnStopFlow'),
            title: "Stop flow",
            intro: 'After testing, the user should stop the flow.',
            position: 'right'
        }]
    }).start().oncomplete(function () {
        setCookie("tourAutomationComplete", true, 90);
    });
}