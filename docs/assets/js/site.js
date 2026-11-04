// Mobile menu and copy buttons. Everything here responds to the visitor; nothing runs on its own.
(function () {
  "use strict";

  // ------------------------------------------------------------ mobile menu
  function initMenu() {
    var toggle = document.querySelector(".nav-toggle");
    var nav = document.getElementById("site-nav");
    if (!toggle || !nav) return;
    toggle.hidden = false;

    function setOpen(open) {
      toggle.setAttribute("aria-expanded", String(open));
      nav.classList.toggle("is-open", open);
    }
    toggle.addEventListener("click", function () {
      setOpen(toggle.getAttribute("aria-expanded") !== "true");
    });
    nav.addEventListener("click", function (event) {
      if (event.target.closest("a")) setOpen(false);
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && nav.classList.contains("is-open")) {
        setOpen(false);
        toggle.focus();
      }
    });
  }

  // ----------------------------------------------------------- copy buttons
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      var ok = document.execCommand("copy");
      document.body.removeChild(area);
      ok ? resolve() : reject(new Error("copy failed"));
    });
  }

  function addCopyButton(pre) {
    var wrapper = pre.parentElement;
    if (!wrapper || wrapper.querySelector(".copy-button")) return;
    var button = document.createElement("button");
    button.type = "button";
    button.className = "copy-button";
    button.textContent = "Copy";
    var status = document.createElement("span");
    status.className = "visually-hidden";
    status.setAttribute("aria-live", "polite");
    var timer = null;

    button.addEventListener("click", function () {
      copyText(pre.textContent).then(
        function () { report("Copied"); },
        function () { report("Copy failed"); }
      );
    });
    function report(message) {
      button.textContent = message;
      status.textContent = message;
      clearTimeout(timer);
      timer = setTimeout(function () {
        button.textContent = "Copy";
        status.textContent = "";
      }, 2000);
    }

    var head = wrapper.querySelector(".code-head");
    if (head) head.appendChild(button);
    else wrapper.insertBefore(button, pre);
    wrapper.appendChild(status);
  }

  window.BERG = window.BERG || {};
  window.BERG.addCopyButtons = function (root) {
    (root || document).querySelectorAll(".code-block pre[data-copy]").forEach(addCopyButton);
  };

  document.addEventListener("DOMContentLoaded", function () {
    initMenu();
    window.BERG.addCopyButtons(document);
  });
})();
