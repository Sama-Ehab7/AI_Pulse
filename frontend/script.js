console.log("JS Loaded");
async function analyze() {
    console.log("Button clicked");

    let text = document.getElementById("inputText").value;

    console.log("Text:", text);

    let response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
    });

    let data = await response.json();

    console.log("Response:", data);

    let resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";

    for (let aspect of data.aspects) {
        let sentiment = data.aspect_sentiments[aspect];

        let div = document.createElement("div");
        div.className = "card " + sentiment;
        div.innerText = aspect + " → " + sentiment;

        resultsDiv.appendChild(div);
    }
}