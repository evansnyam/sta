chrome.runtime.onInstalled.addListener(() => {
  console.log("Extension installed!");
});

chrome.contextMenus.create({
  id: "check-cyberbully",
  title: "Detect Cyberbullying",
  contexts: ["selection"],
});

chrome.contextMenus.onClicked.addListener(async function (info, tab) {
  if (info.menuItemId === "check-cyberbully") {
    const formData = new FormData();
    formData.append("user_input", info.selectionText);

    const response = await fetch(
      "https://detection-main-production.up.railway.app/detect",
      {
        method: "POST",
        body: formData,
      }
    );
    const data = await response.json();
    const isOffensive = data?.details?.offensive;
    const reasons = data?.details?.offensive_reasons;
    const multiClass = data?.details?.multi_class_result;

    // open pop up
    chrome.windows.create({
      url: "popup.html",
      type: "popup",
      width: 700,
      height: 800,
      top: 100,
      left: 100,
    });

    setTimeout(async () => {
      const [tab] = await chrome.tabs.query({
        active: true,
        lastFocusedWindow: true,
      });
      console.log({ tab });
      const resp = await chrome.tabs.sendMessage(tab.id, {
        isOffensive,
        reasons,
        multiClass,
      });
      console.log({ resp });
    }, 3000);
  }
});

function showPopUp(data) {
  console.log(data.details);
  const reasons = data.details.offensive_reasons;
  let reasonsList = "";
  if (reasons.length > 0) {
    reasonsList = "<ul>";
    reasons.forEach((reason) => {
      reasonsList += `<li>${reason}</li>`;
    });
    reasonsList += "</ul>";
  }

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting
      .executeScript({
        target: { tabId: tabs[0].id, allFrames: true },
        func: (reasonsList, data) =>
          document.body.insertAdjacentHTML(
            "afterbegin",
            `<div id="alert-offensive" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(255, 255, 255, 0.9); z-index: 9999;">
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background-color: #3498db; color:white; padding: 20px; box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.2); border-radius: 10px; width: 700px;">
                      <h2 style="margin-bottom: 10px;font-size:2rem; font-weight: bold;">Offensive Content Found on This Page!</h2>
                      <p style="font-weight: 600; margin-top: 2rem;">Reasons:</p>
                      ${reasonsList}
                      <p style="font-weight: 600; margin-top: 2rem;">This content may be offensive due to its association with the following:</p>
                      <span>${data.details.multi_class_result}</span>

                      <p style="font-weight: 600; margin-top: 2rem; font-size: 1.5rem">You can close this tab</p>

                      <div style="display: flex; margin-top: 2rem; align-items:center;">
                      OR <button id="alert-offensive-btn" style="background-color: transparent; padding: 10px 15px; color: white; border-radius: 15px; text-decoration: underline;">Continue Anyway</button>
                      </div>
                  </div>`
          ),
        args: [reasonsList, data],
      })
      // send message "listen-for-close" to content.js
      .then((res) => console.log("injected a function", res))
      .catch((err) => console.log(err));
  });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "sendTextToAPI") {
    const formData = new FormData();
    formData.append("user_input", message.text);
    fetch("https://detection-main-production.up.railway.app/detect", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          console.error(
            "Failed to send text to API",
            response?.status,
            response?.statusText
          );
          throw new Error("Failed to send text to API");
        }
        return response.json();
      })
      .then(async (data) => {
        if (data?.details?.offensive) {
          showPopUp(data);
        } else {
          console.log(`Text from ${message.url} is not offensive!`);
        }
      })
      .catch((error) => {
        console.log("Error sending text to API:", error);
      });
  }
});
