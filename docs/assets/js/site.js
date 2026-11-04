// Colour theme switch, mobile menu and copy buttons. Everything here responds to the visitor; nothing runs on its own.
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

  // ----------------------------------------------------------- colour theme
  // Cycles auto → light → dark, like alegifford.com. "auto" follows the system setting.
  // The saved choice is applied by the inline script in the page head before the page is drawn.
  var THEMES = ["auto", "light", "dark"];
  var THEME_ICONS = { auto: "◐", light: "○", dark: "●" };

  function initTheme() {
    var toggle = document.querySelector(".theme-toggle");
    if (!toggle) return;
    var root = document.documentElement;
    var current = root.getAttribute("data-theme") || "auto";
    if (THEMES.indexOf(current) === -1) current = "auto";

    function show(theme) {
      toggle.querySelector(".theme-icon").textContent = THEME_ICONS[theme];
      toggle.querySelector(".theme-label").textContent = theme;
      var next = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
      toggle.setAttribute("aria-label", "Colour theme: " + theme + ". Switch to " + next + ".");
    }
    toggle.addEventListener("click", function () {
      current = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
      root.setAttribute("data-theme", current);
      try { localStorage.setItem("berg-theme", current); } catch (e) { /* not saved, still applied */ }
      show(current);
    });
    show(current);
    toggle.hidden = false;
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
    initTheme();
    initMenu();
    window.BERG.addCopyButtons(document);
  });
})();
