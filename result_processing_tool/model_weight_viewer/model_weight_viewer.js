"use strict";

const state = {
  checkpoint: null,
  selected: null,
  page: null,
};

const element = (id) => document.getElementById(id);
const dropZone = element("dropZone");
const fileInput = element("fileInput");
const loadStatus = element("loadStatus");
const workspace = element("workspace");
const uploadPanel = element("uploadPanel");
const tensorRows = element("tensorRows");
const valueRows = element("valueRows");

function formatInteger(value) {
  return new Intl.NumberFormat().format(value);
}

function formatBytes(value) {
  if (value === 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index === 0 ? 0 : 2)} ${units[index]}`;
}

function shapeText(shape) {
  return shape.length ? `[${shape.join(", ")}]` : "scalar []";
}

function setStatus(message, kind = "") {
  loadStatus.textContent = message;
  loadStatus.className = `load-status ${kind}`;
}

async function responseJson(response) {
  const payload = await response.json().catch(() => ({ error: "The server returned an invalid response." }));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

async function loadFile(file) {
  if (!file) return;
  setStatus(`Loading ${file.name} (${formatBytes(file.size)})…`, "loading");
  dropZone.disabled = true;
  try {
    const response = await fetch("/api/load", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Filename": encodeURIComponent(file.name),
      },
      body: file,
    });
    const checkpoint = await responseJson(response);
    if (state.checkpoint) {
      fetch(`/api/checkpoint?id=${encodeURIComponent(state.checkpoint.checkpoint_id)}`, { method: "DELETE" });
    }
    state.checkpoint = checkpoint;
    state.selected = null;
    setStatus("");
    renderCheckpoint();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    dropZone.disabled = false;
    fileInput.value = "";
  }
}

function renderCheckpoint() {
  const checkpoint = state.checkpoint;
  uploadPanel.hidden = true;
  workspace.hidden = false;
  element("filename").textContent = checkpoint.filename;
  element("modelType").textContent = checkpoint.model_type || "Not stored";
  element("datasetType").textContent = checkpoint.dataset_type || "Not stored";
  element("tensorCount").textContent = formatInteger(checkpoint.tensor_count);
  element("totalValues").textContent = formatInteger(checkpoint.total_values);
  element("totalBytes").textContent = formatBytes(checkpoint.total_bytes);
  element("sourceFormat").textContent = checkpoint.source_format;
  element("searchInput").value = "";
  renderTensorRows();
  element("inspectorPlaceholder").hidden = false;
  element("inspectorContent").hidden = true;
}

function renderTensorRows() {
  const query = element("searchInput").value.trim().toLowerCase();
  const entries = state.checkpoint.entries.filter((entry) => {
    const haystack = `${entry.name} ${entry.dtype || ""} ${entry.python_type || ""}`.toLowerCase();
    return haystack.includes(query);
  });
  tensorRows.replaceChildren();
  for (const entry of entries) {
    const row = document.createElement("tr");
    row.dataset.name = entry.name;
    if (state.selected && state.selected.name === entry.name) row.classList.add("selected");
    const name = document.createElement("td");
    name.className = "tensor-name";
    name.textContent = entry.name;
    const shape = document.createElement("td");
    shape.className = "shape";
    shape.textContent = entry.kind === "tensor" ? shapeText(entry.shape) : entry.python_type;
    const dtype = document.createElement("td");
    dtype.className = "dtype";
    dtype.textContent = entry.kind === "tensor" ? entry.dtype : "object";
    const count = document.createElement("td");
    count.textContent = entry.kind === "tensor" ? formatInteger(entry.numel) : "—";
    row.append(name, shape, dtype, count);
    if (entry.kind === "tensor") {
      row.tabIndex = 0;
      row.addEventListener("click", () => selectTensor(entry));
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") selectTensor(entry);
      });
    }
    tensorRows.append(row);
  }
  element("visibleCount").textContent = `${entries.length} / ${state.checkpoint.entries.length}`;
  element("emptyState").hidden = entries.length !== 0;
}

async function selectTensor(entry) {
  state.selected = entry;
  state.page = null;
  renderTensorRows();
  element("inspectorPlaceholder").hidden = true;
  element("inspectorContent").hidden = false;
  element("selectedName").textContent = entry.name;
  element("selectedShape").textContent = shapeText(entry.shape);
  element("selectedDtype").textContent = entry.dtype;
  element("selectedNdim").textContent = entry.ndim;
  element("selectedNumel").textContent = formatInteger(entry.numel);
  element("statistics").className = "statistics muted";
  element("statistics").textContent = "Computed only when requested.";
  element("offsetInput").value = 0;
  await loadValuePage();
}

function coordinateFor(flatIndex, shape) {
  if (!shape.length) return "[]";
  let remaining = flatIndex;
  const coordinate = new Array(shape.length);
  for (let index = shape.length - 1; index >= 0; index -= 1) {
    coordinate[index] = remaining % shape[index];
    remaining = Math.floor(remaining / shape[index]);
  }
  return `[${coordinate.join(", ")}]`;
}

async function loadValuePage() {
  if (!state.selected) return;
  const offset = Math.max(0, Number.parseInt(element("offsetInput").value || "0", 10));
  const limit = Number.parseInt(element("pageSize").value, 10);
  const params = new URLSearchParams({
    id: state.checkpoint.checkpoint_id,
    name: state.selected.name,
    offset: String(offset),
    limit: String(limit),
  });
  element("pageRange").textContent = "Loading values…";
  try {
    state.page = await responseJson(await fetch(`/api/tensor?${params}`));
    element("offsetInput").value = state.page.offset;
    renderValuePage();
  } catch (error) {
    element("pageRange").textContent = error.message;
    valueRows.replaceChildren();
  }
}

function renderValuePage() {
  const page = state.page;
  valueRows.replaceChildren();
  page.values.forEach((value, index) => {
    const flatIndex = page.offset + index;
    const row = document.createElement("tr");
    for (const text of [String(flatIndex), coordinateFor(flatIndex, state.selected.shape), value]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.append(cell);
    }
    valueRows.append(row);
  });
  const start = page.returned ? page.offset + 1 : 0;
  const end = page.offset + page.returned;
  element("pageRange").textContent = `${formatInteger(start)}–${formatInteger(end)} of ${formatInteger(page.total)}`;
  element("previousPage").disabled = page.offset === 0;
  element("nextPage").disabled = end >= page.total;
}

async function computeStatistics() {
  if (!state.selected) return;
  const output = element("statistics");
  output.className = "statistics muted";
  output.textContent = "Computing…";
  const params = new URLSearchParams({
    id: state.checkpoint.checkpoint_id,
    name: state.selected.name,
  });
  try {
    const payload = await responseJson(await fetch(`/api/statistics?${params}`));
    output.className = "statistics";
    output.replaceChildren();
    const labels = {
      count: "Count", finite_count: "Finite", minimum: "Minimum", maximum: "Maximum",
      mean: "Mean", standard_deviation: "Std. deviation", zero_count: "Zeros",
      nan_count: "NaN", positive_infinity_count: "+Infinity", negative_infinity_count: "−Infinity",
      value_kind: "Measure",
    };
    for (const [key, value] of Object.entries(payload.statistics)) {
      const item = document.createElement("div");
      const label = document.createElement("span");
      const data = document.createElement("strong");
      label.textContent = labels[key] || key;
      data.textContent = typeof value === "number" ? formatInteger(value) : value;
      item.append(label, data);
      output.append(item);
    }
  } catch (error) {
    output.className = "statistics muted";
    output.textContent = error.message;
  }
}

async function copyCurrentPage() {
  if (!state.page) return;
  const lines = state.page.values.map((value, index) => {
    const flatIndex = state.page.offset + index;
    return `${flatIndex}\t${coordinateFor(flatIndex, state.selected.shape)}\t${value}`;
  });
  await navigator.clipboard.writeText(["flat_index\tcoordinate\tvalue", ...lines].join("\n"));
  const button = element("copyButton");
  const original = button.textContent;
  button.textContent = "Copied";
  setTimeout(() => { button.textContent = original; }, 1000);
}

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => loadFile(fileInput.files[0]));

// Prevent the browser from navigating to or downloading a file when it is
// released anywhere on the page. The drop zone handles the actual upload.
for (const eventName of ["dragenter", "dragover"]) {
  document.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  });
}
document.addEventListener("drop", (event) => {
  const handledByDropZone = event.defaultPrevented;
  event.preventDefault();
  if (!handledByDropZone && event.dataTransfer && event.dataTransfer.files.length) {
    loadFile(event.dataTransfer.files[0]);
  }
});

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}
dropZone.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));
element("loadAnother").addEventListener("click", () => {
  workspace.hidden = true;
  uploadPanel.hidden = false;
  setStatus("");
});
element("searchInput").addEventListener("input", renderTensorRows);
element("statsButton").addEventListener("click", computeStatistics);
element("pageSize").addEventListener("change", loadValuePage);
element("offsetInput").addEventListener("change", loadValuePage);
element("previousPage").addEventListener("click", () => {
  element("offsetInput").value = Math.max(0, state.page.offset - state.page.limit);
  loadValuePage();
});
element("nextPage").addEventListener("click", () => {
  element("offsetInput").value = state.page.offset + state.page.returned;
  loadValuePage();
});
element("copyButton").addEventListener("click", copyCurrentPage);
