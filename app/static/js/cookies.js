
function setCookie(cname, cvalue, exdays) {
    const d = new Date();
    d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
    let expires = "expires=" + d.toUTCString();
    document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/";
}

function getCookie(cname) {
    let name = cname + "=";
    let decodedCookie = decodeURIComponent(document.cookie);
    let ca = decodedCookie.split(';');
    for (let i = 0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return "";
}

function RestartTour() {
    setCookie("tourIndexComplete", false, 90);
    setCookie("tourDataStudioPart1Complete", false, 90);
    setCookie("tourDataStudioPart2Complete", false, 90);
    setCookie("tourTrainComplete", true, 90);
}

function SelectTours(e) {
    var a = e.checked;
    if (a == true) {
        RestartTour();
    } else {
        //Disable all tours
        setCookie("tourIndexComplete", true, 90);
        setCookie("tourDataStudioPart1Complete", true, 90);
        setCookie("tourDataStudioPart2Complete", true, 90);
        setCookie("tourTrainComplete", true, 90);
    }
}
