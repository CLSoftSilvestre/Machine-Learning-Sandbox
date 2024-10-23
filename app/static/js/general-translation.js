var tr = new Translate("EN");

readLanguage();

function readLanguage(){
    var language = localStorage.getItem("language");

    switch (language) {
        case "EN":
            console.log("Lingua selecionada = English");  
            document.getElementById("selectedFlag").src = "../static/img/flags/uk.png";
            translateMenu(language);
            break;
        case "PT":
            document.getElementById("selectedFlag").src = "../static/img/flags/portugal.png";
            console.log("Lingua selecionada = Portugues");  
            translateMenu(language);
            break;
        case "FR":
            document.getElementById("selectedFlag").src = "../static/img/flags/flag.png";
            console.log("Lingua selecionada = French");
            translateMenu(language);
            break;

        default:
            localStorage.setItem("language", "EN");
            break;
    }
}

function setLanguage(lg){
    localStorage.setItem("language", lg);
    //readLanguage();
    //translateMenu(lg);
    location.reload();
}

function translateMenu(lg){
    var listElements = document.querySelectorAll("[id^=tr_]");
    listElements.forEach(element => {
        try {
            element.innerHTML = tr.getTranslation(element.id, lg);
            //console.log("Element: " + element.id + " -> Translation: " + tr.getTranslation(element.id, lg));
        } catch (error) {
            console.log("Error translating " + element.id);
        }
    });
}

