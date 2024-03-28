chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  console.log("Received request in content JS:", request);
  const resultContainer = document.getElementById("result-container");
  const h1 = document.createElement("h1");
  h1.innerText = request.isOffensive
    ? "The text is offensive"
    : "The text is not offensive";
  resultContainer.appendChild(h1);

  const ul = document.createElement("ul");
  request.reasons.forEach((reason) => {
    const li = document.createElement("li");
    li.innerText = reason;
    ul.appendChild(li);
  });
  resultContainer.appendChild(ul);

  if (request?.multiClass) {
    // add p tag to show multi class result
    const p = document.createElement("p");
    p.innerText = request.multiClass;
    resultContainer.appendChild(p);
  }
  console.log("DONE");

  sendResponse({ isOffensive: request.isOffensive, reasons: request.reasons });
});
