(function () {
  const ZOOM_MIN = 0.2;
  const ZOOM_MAX = 3.0;
  const ZOOM_STEP = 0.1;

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function setupMermaidZoom(root) {
    if (root.dataset.mermaidZoom === "1") {
      return;
    }

    const svg = root.querySelector("svg");
    if (!svg) {
      return;
    }

    root.dataset.mermaidZoom = "1";

    const wrapper = document.createElement("div");
    wrapper.className = "mermaid-zoom-wrapper";

    const controls = document.createElement("div");
    controls.className = "mermaid-zoom-controls";

    const zoomIn = document.createElement("button");
    zoomIn.type = "button";
    zoomIn.textContent = "+";

    const zoomOut = document.createElement("button");
    zoomOut.type = "button";
    zoomOut.textContent = "-";

    const reset = document.createElement("button");
    reset.type = "button";
    reset.textContent = "Reset";

    controls.appendChild(zoomIn);
    controls.appendChild(zoomOut);
    controls.appendChild(reset);

    const container = document.createElement("div");
    container.className = "mermaid-zoom-container";

    const stage = document.createElement("div");
    stage.className = "mermaid-zoom-stage";

    root.innerHTML = "";
    stage.appendChild(svg);
    container.appendChild(stage);
    wrapper.appendChild(controls);
    wrapper.appendChild(container);
    root.appendChild(wrapper);

    let scale = 1;
    let offsetX = 0;
    let offsetY = 0;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;

    function applyTransform() {
      stage.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
    }

    function zoomAt(delta, clientX, clientY) {
      const rect = container.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;

      const prevScale = scale;
      scale = clamp(scale + delta, ZOOM_MIN, ZOOM_MAX);
      if (scale === prevScale) {
        return;
      }

      const k = scale / prevScale;
      offsetX = x - k * (x - offsetX);
      offsetY = y - k * (y - offsetY);
      applyTransform();
    }

    zoomIn.addEventListener("click", () => {
      zoomAt(ZOOM_STEP, container.clientWidth / 2, container.clientHeight / 2);
    });

    zoomOut.addEventListener("click", () => {
      zoomAt(-ZOOM_STEP, container.clientWidth / 2, container.clientHeight / 2);
    });

    reset.addEventListener("click", () => {
      scale = 1;
      offsetX = 0;
      offsetY = 0;
      applyTransform();
    });

    container.addEventListener("wheel", (event) => {
      if (!event.ctrlKey && !event.metaKey) {
        return;
      }
      event.preventDefault();
      const delta = event.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
      zoomAt(delta, event.clientX, event.clientY);
    });

    container.addEventListener("mousedown", (event) => {
      if (event.button !== 0) {
        return;
      }
      isDragging = true;
      dragStartX = event.clientX - offsetX;
      dragStartY = event.clientY - offsetY;
      container.classList.add("is-dragging");
    });

    window.addEventListener("mousemove", (event) => {
      if (!isDragging) {
        return;
      }
      offsetX = event.clientX - dragStartX;
      offsetY = event.clientY - dragStartY;
      applyTransform();
    });

    window.addEventListener("mouseup", () => {
      if (!isDragging) {
        return;
      }
      isDragging = false;
      container.classList.remove("is-dragging");
    });

    applyTransform();
  }

  function init() {
    const candidates = document.querySelectorAll(
      "pre.mermaid, div.mermaid, .mermaid-container"
    );
    candidates.forEach(setupMermaidZoom);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
