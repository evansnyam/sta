console.log("Content script loaded");

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  return false;
});

document.addEventListener("DOMContentLoaded", () => {
  console.log("Content script loaded");
  const pageText = document.body.innerText.trim();
  chrome.runtime.sendMessage({ action: "sendTextToAPI", text: pageText });
});

function getText() {
  return document.body.innerText;
}
function getHTML() {
  return document.body.outerHTML;
}
function getURL() {
  return window.location.href;
}

// send url, text, and html to background.js
chrome.runtime.sendMessage({
  action: "sendTextToAPI",
  text: getText(),
  html: getHTML(),
  url: getURL(),
});

// listen for click on button with id "alert-offensive-btn" then run document.getElementById('alert-offensive')?.remove()

/**
 * Wait for an element before resolving a promise
 * @param {String} querySelector - Selector of element to wait for
 * @param {Integer} timeout - Milliseconds to wait before timing out, or 0 for no timeout
 */
function waitForElement(querySelector, timeout) {
  return new Promise((resolve, reject) => {
    var timer = false;
    if (document.querySelectorAll(querySelector).length) return resolve();
    const observer = new MutationObserver(() => {
      if (document.querySelectorAll(querySelector).length) {
        observer.disconnect();
        if (timer !== false) clearTimeout(timer);
        return resolve();
      }
    });
    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
    if (timeout)
      timer = setTimeout(() => {
        observer.disconnect();
        reject();
      }, timeout);
  });
}

waitForElement("#alert-offensive", 10000)
  .then(function () {
    document
      .getElementById("alert-offensive-btn")
      .addEventListener("click", () => {
        document.getElementById("alert-offensive")?.remove();
      });
  })
  .catch(() => {
    console.log("Timed out waiting for element");
  });
