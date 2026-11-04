// Model directory: adds search and three pill filters to the static tables written by
// scripts/website/build_catalog.py, using the catalog embedded in the page.
// Without JavaScript, or if the catalog can't be read, the tables stay as they are.
(function () {
  "use strict";

  var FILTERS = [
    { name: "modality", vocab: "modality", values: function (m) { return [m.modality]; } },
    { name: "stimulus", vocab: "stimulus_types", values: function (m) { return m.stimulus_types; } },
    { name: "species", vocab: "species", values: function (m) { return [m.species]; } }
  ];

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === "text") node.textContent = attrs[key];
      else if (key === "className") node.className = attrs[key];
      else node.setAttribute(key, attrs[key]);
    });
    (children || []).forEach(function (child) { if (child) node.appendChild(child); });
    return node;
  }

  function labelMap(entries) {
    var map = {};
    entries.forEach(function (e) { map[e.key] = e.label; });
    return map;
  }

  function Explorer(root, catalog) {
    this.root = root;
    this.catalog = catalog;
    this.models = catalog.models;
    this.labels = {
      modality: labelMap(catalog.vocabulary.modality),
      species: labelMap(catalog.vocabulary.species),
      stimulus_types: labelMap(catalog.vocabulary.stimulus_types),
      group: labelMap(catalog.groups)
    };
    this.state = { modality: null, stimulus: null, species: null, q: "" };
    this.controls = root.querySelector(".explorer-controls");
    this.search = root.querySelector("#model-search");
    this.count = root.querySelector(".explorer-count");
    this.sections = root.querySelectorAll(".directory-section");
    this.empty = root.querySelector(".explorer-empty");
    this.rows = [];
  }

  Explorer.prototype.init = function () {
    if (!this.collectRows()) return;
    this.readUrl();
    this.renderFilters();
    this.bind();
    this.controls.hidden = false;
    this.apply(false);
  };

  // Pair each static table row with its catalog entry. Gives up if they disagree.
  Explorer.prototype.collectRows = function () {
    var byId = {};
    this.models.forEach(function (m) { byId[m.id] = m; });
    var self = this;
    var ok = true;
    this.root.querySelectorAll("tr.model").forEach(function (row) {
      var m = byId[row.id.replace(/^model-/, "")];
      if (!m) { ok = false; return; }
      row._model = m;
      row._haystack = [
        m.title, m.dataset, m.architecture, m.id, m.dataset_full, m.architecture_full, m.stimulus_label,
        self.labels.group[m.group], self.labels.modality[m.modality], self.labels.species[m.species],
        m.variant_note || ""
      ].join(" ").toLowerCase();
      self.addUsage(row, m);
      self.rows.push(row);
    });
    return ok && this.rows.length === this.models.length;
  };

  // Verified snippets get a Usage toggle that opens a row with the code below the model.
  Explorer.prototype.addUsage = function (row, m) {
    if (!m.snippet_verified || !m.snippet) return;
    var usageId = "usage-" + m.id;
    var toggle = el("button", {
      type: "button",
      className: "link-button usage-toggle",
      "aria-expanded": "false",
      "aria-controls": usageId
    }, [document.createTextNode("Usage"), el("span", { className: "visually-hidden", text: " of " + m.id })]);
    var usage = el("tr", { className: "model-usage-row", id: usageId }, [
      el("td", { colspan: "5" }, [
        el("div", { className: "code-block" }, [
          el("div", { className: "code-head", text: "Python" }),
          el("pre", { "data-copy": "" }, [el("code", { className: "language-python", text: m.snippet })])
        ])
      ])
    ]);
    usage.hidden = true;
    row.querySelector(".model-links").appendChild(toggle);
    row.parentNode.insertBefore(usage, row.nextSibling);
    toggle.addEventListener("click", function () {
      var open = toggle.getAttribute("aria-expanded") !== "true";
      toggle.setAttribute("aria-expanded", String(open));
      usage.hidden = !open;
      if (open && window.BERG && window.BERG.addCopyButtons) window.BERG.addCopyButtons(usage);
    });
    row._usage = usage;
    row._toggle = toggle;
  };

  // --------------------------------------------------------------- URL state
  Explorer.prototype.readUrl = function () {
    var params = new URLSearchParams(window.location.search);
    var self = this;
    FILTERS.forEach(function (f) {
      var value = params.get(f.name);
      if (value && self.optionsFor(f).indexOf(value) !== -1) self.state[f.name] = value;
    });
    this.state.q = params.get("q") || "";
    this.search.value = this.state.q;
  };

  Explorer.prototype.writeUrl = function () {
    var params = new URLSearchParams();
    var state = this.state;
    FILTERS.forEach(function (f) { if (state[f.name]) params.set(f.name, state[f.name]); });
    if (state.q.trim()) params.set("q", state.q.trim());
    var query = params.toString();
    var url = window.location.pathname + (query ? "?" + query : "") + window.location.hash;
    window.history.replaceState(null, "", url);
  };

  // ----------------------------------------------------------------- filters
  // Options come from the models actually in the catalog, in vocabulary order.
  Explorer.prototype.optionsFor = function (filter) {
    var used = {};
    this.models.forEach(function (m) { filter.values(m).forEach(function (v) { used[v] = true; }); });
    return this.catalog.vocabulary[filter.vocab]
      .map(function (e) { return e.key; })
      .filter(function (key) { return used[key]; });
  };

  Explorer.prototype.renderFilters = function () {
    var self = this;
    FILTERS.forEach(function (f) {
      var container = self.root.querySelector('[data-filter="' + f.name + '"]');
      var labels = self.labels[f.vocab];
      [null].concat(self.optionsFor(f)).forEach(function (key) {
        var button = el("button", {
          type: "button",
          className: "pill",
          "data-value": key || "",
          "aria-pressed": String(self.state[f.name] === key),
          text: key ? labels[key] : "All"
        });
        button.addEventListener("click", function () { self.setFilter(f.name, key); });
        container.appendChild(button);
      });
    });
  };

  Explorer.prototype.setFilter = function (name, key) {
    this.state[name] = key;
    this.root.querySelectorAll('[data-filter="' + name + '"] .pill').forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-value") === (key || "")));
    });
    this.apply(true);
  };

  // ------------------------------------------------------------------ filter
  Explorer.prototype.matches = function (row, terms) {
    var m = row._model;
    var state = this.state;
    for (var i = 0; i < FILTERS.length; i++) {
      var f = FILTERS[i];
      if (state[f.name] && f.values(m).indexOf(state[f.name]) === -1) return false;
    }
    return terms.every(function (t) { return row._haystack.indexOf(t) !== -1; });
  };

  Explorer.prototype.apply = function (updateUrl) {
    var self = this;
    var terms = this.state.q.toLowerCase().split(/\s+/).filter(Boolean);
    var shown = 0;
    this.rows.forEach(function (row) {
      var visible = self.matches(row, terms);
      row.hidden = !visible;
      if (row._usage) row._usage.hidden = !visible || row._toggle.getAttribute("aria-expanded") !== "true";
      if (visible) shown += 1;
    });
    // Hide a whole section (heading and table) when none of its models match.
    this.sections.forEach(function (section) {
      section.hidden = !section.querySelector("tr.model:not([hidden])");
    });
    var total = this.rows.length;
    var noun = total === 1 ? " model" : " models";
    this.count.textContent = shown === total ? total + noun : shown + " of " + total + noun;
    this.empty.hidden = shown !== 0;
    if (updateUrl) this.writeUrl();
  };

  Explorer.prototype.reset = function () {
    var self = this;
    FILTERS.forEach(function (f) { self.setFilter(f.name, null); });
    this.state.q = "";
    this.search.value = "";
    this.apply(true);
    this.search.focus();
  };

  Explorer.prototype.bind = function () {
    var self = this;
    this.search.addEventListener("input", function () {
      self.state.q = self.search.value;
      self.apply(true);
    });
    var reset = this.root.querySelector('[data-action="reset"]');
    if (reset) reset.addEventListener("click", function () { self.reset(); });
  };

  // The catalog embedded by build_catalog.py works everywhere, including index_new.html opened from disk.
  function embeddedCatalog() {
    var node = document.getElementById("catalog-data");
    if (!node) return null;
    try { return JSON.parse(node.textContent); } catch (e) { return null; }
  }

  document.addEventListener("DOMContentLoaded", function () {
    var root = document.querySelector(".explorer");
    var catalog = embeddedCatalog();
    if (root && catalog && window.URLSearchParams) new Explorer(root, catalog).init();
  });
})();
