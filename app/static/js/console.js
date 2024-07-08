//import Ollama from "ollama-js-client";
const Ollama = window.OllamaJS;

function openForm() {
    document.getElementById("myForm").style.display = "block";
  }
  
function closeForm() {
    document.getElementById("myForm").style.display = "none";
}

function setup(){
  const input = document.getElementById("prompt")
  const output = document.getElementById("consoleOutput")
  const sendButton = document.getElementById("sendBtn")
  const ollamaEndpoint = document.getElementById("OllamaEndpoint").innerHTML
  const ollamaModel = document.getElementById("OllamaModel").innerHTML

  console.log(ollamaEndpoint)
  console.log(ollamaModel)

  const lama = new Ollama({
    model: String(ollamaModel),
    url: String(ollamaEndpoint)
  });

  const on_response = (error,response) => {
    if (error){
      console.error(error)
      output.innerHTML += String(error.error) + '\n';
      logo = document.getElementById("sendLogo");
      logo.className = logo.className === "fa-solid fa-rotate" ? "fa-solid fa-paper-plane" : "fa-solid fa-rotate";
    } else if (response.done){
      logo = document.getElementById("sendLogo");
      logo.className = logo.className === "fa-solid fa-rotate" ? "fa-solid fa-paper-plane" : "fa-solid fa-rotate";
      output.scrollTop = output.scrollHeight;
    } else {
      console.log(response)
      output.innerHTML += String(response.response)
      output.scrollTop = output.scrollHeight;
    }
  }

  sendButton.addEventListener("click", async () =>{

    output.innerHTML += '\n' + "[USER] : " + input.value + '\n';
    output.innerHTML += "[ASSISTANT] : ";
    output.scrollTop = output.scrollHeight;

    // Change the logo of the button
    logo = document.getElementById("sendLogo");
    logo.className = logo.className === "fa-solid fa-paper-plane" ? "fa-solid fa-rotate" : "fa-solid fa-paper-plane";

    await lama.prompt_stream(input.value, on_response)
    input.value = ""

  })

}

setup();
