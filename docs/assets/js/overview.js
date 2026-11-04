// Overview figure: example images and the in silico fMRI responses BERG generates for them.
// Reads the data that scripts/website/make_overview_figure.py embeds in the page, draws the
// bar chart and steps through the images. Without JavaScript the static picture stays.
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var ADVANCE_MS = 4500;
  var TWEEN_MS = 650;
  var PLURAL = { object: "objects", body: "bodies", scene: "scenes", face: "faces" };

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

  function svg(tag, attrs, parent) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === "text") node.textContent = attrs[key];
      else node.setAttribute(key, attrs[key]);
    });
    if (parent) parent.appendChild(node);
    return node;
  }

  function niceStep(span, ticks) {
    var raw = span / ticks;
    var power = Math.pow(10, Math.floor(Math.log10(raw)));
    var f = raw / power;
    return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * power;
  }

  function format(value, step) {
    var digits = step < 0.1 ? 2 : step < 1 ? 1 : 0;
    return value.toFixed(digits).replace("-", "−");
  }

  function ease(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

  // Bar from the zero line to the value, rounded at the data end only.
  function barPath(x, width, y0, y1) {
    var h = Math.abs(y1 - y0);
    var r = Math.min(4, width / 2, h);
    if (h < 0.5) return "M" + x + " " + y0 + "h" + width + "v0h" + (-width) + "z";
    if (y1 < y0) {   // positive value, bar grows upwards
      return "M" + x + " " + y0 + "V" + (y1 + r) + "Q" + x + " " + y1 + " " + (x + r) + " " + y1 +
        "H" + (x + width - r) + "Q" + (x + width) + " " + y1 + " " + (x + width) + " " + (y1 + r) + "V" + y0 + "Z";
    }
    return "M" + x + " " + y0 + "V" + (y1 - r) + "Q" + x + " " + y1 + " " + (x + r) + " " + y1 +
      "H" + (x + width - r) + "Q" + (x + width) + " " + y1 + " " + (x + width) + " " + (y1 - r) + "V" + y0 + "Z";
  }

  function Overview(figure, data) {
    this.figure = figure;
    this.data = data;
    this.images = data.images;
    this.areas = data.areas;
    this.placeholder = !!data.placeholder;
    this.index = 0;
    this.shown = this.target(0);
    this.reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.userPaused = false;
    this.hovered = false;
    this.focused = false;
    this.visible = true;
    this.timer = null;
    this.frame = null;
  }

  Overview.prototype.init = function () {
    var f = this.figure;
    this.stage = f.querySelector(".stimulus-stage");
    this.picker = f.querySelector(".stimulus-picker");
    this.chart = f.querySelector(".response-chart");
    this.toggle = f.querySelector(".overview-toggle");
    this.status = f.querySelector(".overview-status");
    f.classList.add("is-live");      // before drawing, so the chart has a width to measure
    this.buildStimuli();
    this.buildTable();
    this.draw();
    this.bind();
    this.update();
  };

  // Values on screen for image i: the mean per area, then each subject per area.
  Overview.prototype.target = function (i) {
    var image = this.images[i];
    var flat = image.values.slice();
    (image.subjects || []).forEach(function (row) { flat = flat.concat(row); });
    return flat.map(function (v) { return v === null || v === undefined ? 0 : v; });
  };

  // ---------------------------------------------------------------- stimuli
  Overview.prototype.buildStimuli = function () {
    var self = this;
    this.stageImages = this.images.map(function (image, i) {
      var img = el("img", { src: image.src, alt: image.alt, decoding: "async" });
      if (i) img.setAttribute("aria-hidden", "true");
      img.classList.toggle("is-active", i === 0);
      img.classList.toggle("is-whole", image.fit === "contain");   // shown uncropped, e.g. for CC BY-ND photos
      self.stage.appendChild(img);
      return img;
    });
    this.buttons = this.images.map(function (image, i) {
      var button = el("button", { type: "button", className: "pick", "aria-pressed": String(i === 0) }, [
        el("img", { src: image.src, alt: "", loading: "lazy", decoding: "async", "class": image.fit === "contain" ? "is-whole" : "" }),
        el("span", { text: image.label })
      ]);
      button.addEventListener("click", function () {
        self.userPaused = true;
        self.select(i, true);
      });
      self.picker.appendChild(button);
      return button;
    });
  };

  Overview.prototype.select = function (i, announce) {
    var self = this;
    this.index = i;
    this.stageImages.forEach(function (img, k) {
      img.classList.toggle("is-active", k === i);
      if (k === i) img.removeAttribute("aria-hidden"); else img.setAttribute("aria-hidden", "true");
    });
    this.buttons.forEach(function (b, k) { b.setAttribute("aria-pressed", String(k === i)); });
    if (announce && this.status) {
      this.status.textContent = "Showing in silico responses to the " + this.images[i].label.toLowerCase() + " image.";
    }
    this.markMatch();
    this.animateTo(this.target(i));
    this.update();
  };

  // ------------------------------------------------------------------ chart
  Overview.prototype.scale = function () {
    var all = [];
    this.images.forEach(function (image) {
      all = all.concat(image.values);
      (image.subjects || []).forEach(function (row) { all = all.concat(row); });
    });
    all = all.filter(function (v) { return v !== null && v !== undefined; });
    var lo = Math.min(0, Math.min.apply(null, all.length ? all : [0]));
    var hi = Math.max(0, Math.max.apply(null, all.length ? all : [1]));
    if (hi - lo === 0) hi = 1;
    var step = niceStep(hi - lo, 4);
    return { lo: Math.floor(lo / step + 1e-9) * step, hi: Math.ceil(hi / step - 1e-9) * step, step: step };
  };

  Overview.prototype.draw = function () {
    var self = this;
    var width = Math.max(240, this.chart.clientWidth);
    var height = Math.round(Math.max(240, Math.min(330, width * 0.72)));
    var m = { top: 30, right: 4, bottom: 46, left: 40 };
    var plotW = width - m.left - m.right;
    var plotH = height - m.top - m.bottom;
    var n = this.areas.length;
    var firstCategory = this.areas.findIndex(function (a) { return !!a.category; });
    var gap = firstCategory > 0 ? 14 : 0;
    var band = (plotW - gap) / n;
    var barW = Math.min(30, Math.max(8, band * 0.6));
    var s = this.scale();
    var y = function (v) { return m.top + (s.hi - v) / (s.hi - s.lo) * plotH; };
    var cx = function (i) { return m.left + band * (i + 0.5) + (firstCategory > 0 && i >= firstCategory ? gap : 0); };

    var root = svg("svg", { width: width, height: height, viewBox: "0 0 " + width + " " + height, "aria-hidden": "true" });
    svg("text", { x: 0, y: 12, "class": "chart-units", text: this.data.units || "In silico response" }, root);
    for (var t = s.lo; t <= s.hi + s.step / 2; t += s.step) {
      var ty = Math.round(y(t)) + 0.5;
      svg("line", { x1: m.left, x2: width - m.right, y1: ty, y2: ty, "class": Math.abs(t) < s.step / 2 ? "chart-zero" : "chart-grid" }, root);
      svg("text", { x: m.left - 8, y: ty + 4, "text-anchor": "end", "class": "chart-tick", text: format(Math.abs(t) < s.step / 2 ? 0 : t, s.step) }, root);
    }

    var labelY = m.top + plotH + 18;
    var roomy = band >= 46;            // narrow bars: no category words under the labels
    this.labels = this.areas.map(function (area, i) {
      var label = svg("text", { x: cx(i), y: labelY, "text-anchor": "middle", "class": "area-label", text: area.key }, root);
      if (area.category && roomy) {
        svg("text", { x: cx(i), y: labelY + 16, "text-anchor": "middle", "class": "area-sub", text: PLURAL[area.category] || area.category }, root);
      }
      return label;
    });
    if (firstCategory > 0 && roomy) {
      svg("text", { x: (cx(0) + cx(firstCategory - 1)) / 2, y: labelY + 16, "text-anchor": "middle", "class": "area-sub", text: "early visual" }, root);
    }

    var nSubjects = (this.images[0].subjects || []).length;
    this.bars = this.areas.map(function (area, i) {
      return svg("path", { "class": "bar" }, root);
    });
    this.dots = [];
    for (var sIdx = 0; sIdx < nSubjects; sIdx++) {
      for (var i = 0; i < n; i++) {
        var offset = (sIdx - (nSubjects - 1) / 2) * Math.min(7, barW / (nSubjects + 0.5));
        this.dots.push(svg("circle", { "class": "dot", cx: cx(i) + offset, r: 2.75 }, root));
      }
    }

    if (this.placeholder) {
      svg("text", { x: m.left + plotW / 2, y: m.top + plotH / 2, "text-anchor": "middle", "class": "chart-tick", text: "Placeholder: no prediction yet" }, root);
    }

    this.geometry = { y: y, cx: cx, barW: barW, n: n, band: band, m: m, plotH: plotH, width: width };
    this.chart.querySelectorAll("svg").forEach(function (old) { old.remove(); });
    this.chart.insertBefore(root, this.chart.firstChild);
    this.markMatch();
    this.render(this.shown);
  };

  Overview.prototype.markMatch = function () {
    var key = this.images[this.index].key;
    var areas = this.areas;
    (this.bars || []).forEach(function (bar, i) { bar.setAttribute("class", areas[i].category === key ? "bar is-match" : "bar"); });
    (this.labels || []).forEach(function (label, i) { label.setAttribute("class", areas[i].category === key ? "area-label is-match" : "area-label"); });
  };

  Overview.prototype.render = function (values) {
    if (this.placeholder) return;
    var g = this.geometry;
    var zero = g.y(0);
    for (var i = 0; i < g.n; i++) {
      this.bars[i].setAttribute("d", barPath(g.cx(i) - g.barW / 2, g.barW, zero, g.y(values[i])));
    }
    for (var k = 0; k < this.dots.length; k++) this.dots[k].setAttribute("cy", g.y(values[g.n + k]));
  };

  Overview.prototype.animateTo = function (target) {
    var self = this;
    var from = this.shown.slice();
    var finish = function () {
      cancelAnimationFrame(self.frame);
      clearTimeout(self.guard);
      self.shown = target;
      self.render(target);
    };
    cancelAnimationFrame(this.frame);
    clearTimeout(this.guard);
    if (this.reduced) {
      finish();
      return;
    }
    var start = null;
    var step = function (now) {
      if (start === null) start = now;
      var t = Math.min(1, (now - start) / TWEEN_MS);
      var e = ease(t);
      self.shown = from.map(function (v, k) { return v + (target[k] - v) * e; });
      self.render(self.shown);
      if (t < 1) self.frame = requestAnimationFrame(step);
      else clearTimeout(self.guard);
    };
    this.frame = requestAnimationFrame(step);
    // If frames stall (background tab, throttled renderer), still end on the right values.
    this.guard = setTimeout(finish, TWEEN_MS + 150);
  };

  // Tooltip with the exact values of the area under the pointer.
  Overview.prototype.bindTooltip = function () {
    var self = this;
    var tip = el("div", { className: "chart-tip", "aria-hidden": "true" });
    tip.hidden = true;
    this.chart.appendChild(tip);
    this.chart.addEventListener("pointermove", function (event) {
      if (self.placeholder) return;
      var g = self.geometry;
      var rect = self.chart.getBoundingClientRect();
      var x = event.clientX - rect.left;
      var best = -1, bestDist = Infinity;
      for (var i = 0; i < g.n; i++) {
        var d = Math.abs(x - g.cx(i));
        if (d < bestDist) { best = i; bestDist = d; }
      }
      if (best < 0 || bestDist > g.band * 0.6) { tip.hidden = true; return; }
      var image = self.images[self.index];
      var area = self.areas[best];
      var step = self.scale().step / 10;
      var perSubject = (image.subjects || []).map(function (row) { return row[best] === null ? "–" : format(row[best], step); });
      tip.textContent = "";
      tip.appendChild(el("strong", { text: area.key + " " }));
      tip.appendChild(document.createTextNode(area.name));
      tip.appendChild(el("br"));
      tip.appendChild(document.createTextNode(
        (image.values[best] === null ? "–" : format(image.values[best], step)) +
        (perSubject.length ? "  (subjects " + perSubject.join(", ") + ")" : "")
      ));
      tip.hidden = false;
      var left = Math.min(Math.max(0, g.cx(best) - tip.offsetWidth / 2), rect.width - tip.offsetWidth);
      tip.style.left = left + "px";
      tip.style.top = Math.max(0, Math.min(g.y(self.shown[best]), g.y(0)) - tip.offsetHeight - 10) + "px";
    });
    this.chart.addEventListener("pointerleave", function () { tip.hidden = true; });
  };

  // Screen readers get every value as a table instead of the drawing.
  Overview.prototype.buildTable = function () {
    if (this.placeholder) return;
    var areas = this.areas;
    var table = el("table", { className: "visually-hidden" }, [
      el("caption", { text: (this.data.units || "In silico response") + " by image and brain area, averaged across subjects" }),
      el("thead", {}, [el("tr", {}, [el("th", { scope: "col", text: "Image" })].concat(
        areas.map(function (a) { return el("th", { scope: "col", text: a.key + " (" + a.name + ")" }); })))]),
      el("tbody", {}, this.images.map(function (image) {
        return el("tr", {}, [el("th", { scope: "row", text: image.label })].concat(
          image.values.map(function (v) { return el("td", { text: v === null ? "–" : v.toFixed(2) }); })));
      }))
    ]);
    this.chart.appendChild(table);
  };

  // --------------------------------------------------------------- playback
  Overview.prototype.bind = function () {
    var self = this;
    var f = this.figure;
    this.bindTooltip();
    if (this.toggle) {
      this.toggle.addEventListener("click", function () {
        self.userPaused = !self.userPaused;
        self.update();
      });
    }
    var flow = f.querySelector(".overview-flow");
    flow.addEventListener("pointerenter", function () { self.hovered = true; self.update(); });
    flow.addEventListener("pointerleave", function () { self.hovered = false; self.update(); });
    f.addEventListener("focusin", function () { self.focused = true; self.update(); });
    f.addEventListener("focusout", function (event) {
      if (!f.contains(event.relatedTarget)) { self.focused = false; self.update(); }
    });
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        self.visible = entries[0].isIntersecting;
        self.update();
      }).observe(f);
    }
    document.addEventListener("visibilitychange", function () { self.update(); });
    // Redraw when the chart's width changes (window resize, layout switch).
    var lastWidth = this.chart.clientWidth;
    var redraw = function () {
      if (self.chart.clientWidth !== lastWidth) { lastWidth = self.chart.clientWidth; self.draw(); }
    };
    if ("ResizeObserver" in window) new ResizeObserver(redraw).observe(this.chart);
    else window.addEventListener("resize", redraw);
  };

  Overview.prototype.canPlay = function () {
    return !this.reduced && !this.placeholder && this.images.length > 1;
  };

  Overview.prototype.update = function () {
    var self = this;
    var playing = this.canPlay() && !this.userPaused && !this.hovered && !this.focused && this.visible && !document.hidden;
    clearTimeout(this.timer);
    if (playing) {
      this.timer = setTimeout(function () { self.select((self.index + 1) % self.images.length, false); }, ADVANCE_MS);
    }
    if (this.toggle) {
      this.toggle.hidden = !this.canPlay();
      this.toggle.setAttribute("aria-pressed", String(this.userPaused));
      this.toggle.querySelector(".label").textContent = this.userPaused ? "Play" : "Pause";
    }
  };

  document.addEventListener("DOMContentLoaded", function () {
    var figure = document.querySelector(".overview");
    var node = document.getElementById("overview-data");
    if (!figure || !node) return;
    var data = null;
    try { data = JSON.parse(node.textContent); } catch (e) { data = null; }
    if (!data || !data.images || data.images.length < 1 || !data.areas) {
      figure.classList.add("is-static");
      return;
    }
    new Overview(figure, data).init();
  });
})();
