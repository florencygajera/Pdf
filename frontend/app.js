const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const fileMeta = document.getElementById("file-meta");
const refreshBtn = document.getElementById("refresh-btn");
const clearBtn = document.getElementById("clear-btn");
const copyBtn = document.getElementById("copy-btn");
const downloadBtn = document.getElementById("download-btn");
const apiHealth = document.getElementById("api-health");
const readyHealth = document.getElementById("ready-health");
const statusText = document.getElementById("status-text");
const jobIdEl = document.getElementById("job-id");
const jobStatusEl = document.getElementById("job-status");
const jobProgressEl = document.getElementById("job-progress");
const progressBar = document.getElementById("progress-bar");
const resultText = document.getElementById("result-text");
const rawJson = document.getElementById("raw-json");
const metadataEl = document.getElementById("metadata");
const tablesEl = document.getElementById("tables");
const tabButtons = Array.from(document.querySelectorAll(".tab"));
const panelViews = {
  text: document.getElementById("panel-text"),
  metadata: document.getElementById("panel-metadata"),
  tables: document.getElementById("panel-tables"),
  raw: document.getElementById("panel-raw"),
};

let currentJobId = "";
let currentResult = null;
let currentText = "";
let activePoll = null;

function setBadge(el, label, state) {
  el.className = `pill ${state}`;
  el.textContent = label;
}

function setProgress(percent) {
  const value = Math.max(0, Math.min(100, Number(percent) || 0));
  progressBar.style.width = `${value}%`;
  jobProgressEl.textContent = `${Math.round(value)}%`;
}

function prettyBytes(bytes) {
  if (!bytes && bytes !== 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
}

function switchTab(tabName) {
  tabButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tabName));
  Object.entries(panelViews).forEach(([name, panel]) => {
    panel.classList.toggle("active", name === tabName);
  });
}

function resetResult() {
  currentResult = null;
  currentText = "";
  currentJobId = "";
  jobIdEl.textContent = "No job yet";
  jobStatusEl.textContent = "Idle";
  setProgress(0);
  setBadge(apiHealth, "Checking API...", "pill-warm");
  statusText.textContent = "Upload a file to begin.";
  resultText.textContent = "Waiting for extraction result...";
  rawJson.textContent = "{}";
  metadataEl.innerHTML = "";
  tablesEl.innerHTML = '<div class="empty-state">No tables yet.</div>';
  fileMeta.textContent = "No file selected.";
}

function renderMetadata(metadata = {}) {
  const entries = [
    ["Pages", metadata.pages ?? "-"],
    ["PDF Type", metadata.pdf_type ?? "-"],
    ["Confidence", metadata.confidence_score ?? "-"],
    ["Processing time", metadata.processing_time_seconds ?? "-"],
    [
      "Languages",
      Array.isArray(metadata.languages_detected) && metadata.languages_detected.length
        ? metadata.languages_detected.join(", ")
        : "-",
    ],
    ["OCR engine", metadata.ocr_engine ?? "-"],
  ];

  metadataEl.innerHTML = entries
    .map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`)
    .join("");
}

function renderTables(tables = []) {
  if (!tables.length) {
    tablesEl.innerHTML = '<div class="empty-state">No tables extracted.</div>';
    return;
  }

  tablesEl.innerHTML = tables
    .map((table, index) => {
      const headers = table.headers ?? [];
      const rows = table.rows ?? [];
      const title = `Table ${index + 1}`;
      const method = table.extraction_method || "unknown";
      const head = headers.length
        ? `<tr>${headers.map((header) => `<th>${header || "&nbsp;"}</th>`).join("")}</tr>`
        : "<tr><th>Data</th></tr>";
      const body = rows.length
        ? rows
            .map(
              (row) =>
                `<tr>${row.map((cell) => `<td>${cell || "&nbsp;"}</td>`).join("")}</tr>`
            )
            .join("")
        : '<tr><td class="empty-state">No rows available.</td></tr>';

      return `
        <article class="table">
          <div class="table-head">
            <strong>${title}</strong>
            <span class="meta">${method}</span>
          </div>
          <table class="table-grid">
            <thead>${head}</thead>
            <tbody>${body}</tbody>
          </table>
        </article>
      `;
    })
    .join("");
}

function renderResult(data) {
  currentResult = data;
  currentText = data?.text || "";
  resultText.textContent = currentText || "No text extracted.";
  rawJson.textContent = JSON.stringify(data || {}, null, 2);
  renderMetadata(data?.metadata || {});
  renderTables(data?.tables || []);
}

async function fetchApiHealth() {
  try {
    const [healthRes, readyRes] = await Promise.all([
      fetch("/healthz"),
      fetch("/readyz"),
    ]);

    if (healthRes.ok) {
      setBadge(apiHealth, "API online", "pill-success");
    } else {
      setBadge(apiHealth, "API issue", "pill-danger");
    }

    if (readyRes.ok) {
      setBadge(readyHealth, "Ready", "pill-success");
    } else {
      setBadge(readyHealth, "Not ready", "pill-warm");
    }
  } catch {
    setBadge(apiHealth, "Offline", "pill-danger");
    setBadge(readyHealth, "Unavailable", "pill-danger");
  }
}

async function refreshStatus() {
  if (!currentJobId) {
    statusText.textContent = "No active job.";
    return;
  }

  try {
    const response = await fetch(`/api/v1/extract/${currentJobId}/status`);
    if (!response.ok) {
      throw new Error(`Status request failed (${response.status})`);
    }
    const data = await response.json();
    jobStatusEl.textContent = data.status || "unknown";
    setProgress(data.progress_percent || 0);
    statusText.textContent = data.message || "Status refreshed.";
  } catch (error) {
    statusText.textContent = error.message || "Failed to refresh status.";
  }
}

async function loadFinalResult(jobId) {
  const response = await fetch(`/api/v1/extract/${jobId}`);
  if (response.status === 202) {
    const body = await response.json();
    jobStatusEl.textContent = body.status || "processing";
    setProgress(body.progress_percent || 0);
    statusText.textContent = body.message || "Processing...";
    return { done: false };
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed (${response.status})`);
  }

  const data = await response.json();
  renderResult(data);
  jobStatusEl.textContent = data.status || "done";
  setProgress(100);
  statusText.textContent = "Extraction complete.";
  switchTab("text");
  return { done: true };
}

function startPolling(jobId) {
  if (activePoll) {
    clearTimeout(activePoll);
  }

  const tick = async () => {
    try {
      const outcome = await loadFinalResult(jobId);
      if (!outcome.done) {
        activePoll = setTimeout(tick, 1200);
      } else {
        activePoll = null;
      }
    } catch (error) {
      jobStatusEl.textContent = "failed";
      statusText.textContent = error.message || "Polling failed.";
      setBadge(apiHealth, "Job error", "pill-danger");
      activePoll = null;
    }
  };

  tick();
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  statusText.textContent = "Uploading file...";
  jobStatusEl.textContent = "uploading";
  setProgress(12);
  setBadge(apiHealth, "Uploading", "pill-warm");

  const response = await fetch("/api/v1/upload", {
    method: "POST",
    body: formData,
  });
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "Upload failed.");
  }

  currentJobId = data.job_id;
  jobIdEl.textContent = currentJobId;
  statusText.textContent = "Upload accepted. Polling extraction job...";
  jobStatusEl.textContent = "queued";
  setProgress(28);
  startPolling(currentJobId);
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  fileMeta.textContent = file
    ? `${file.name} · ${prettyBytes(file.size)}`
    : "No file selected.";
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  fileInput.files = event.dataTransfer.files;
  fileMeta.textContent = `${file.name} · ${prettyBytes(file.size)}`;
});

refreshBtn.addEventListener("click", refreshStatus);

clearBtn.addEventListener("click", () => {
  if (activePoll) {
    clearTimeout(activePoll);
    activePoll = null;
  }
  fileInput.value = "";
  resetResult();
  fetchApiHealth();
});

copyBtn.addEventListener("click", async () => {
  if (!currentText) return;
  try {
    await navigator.clipboard.writeText(currentText);
    copyBtn.textContent = "Copied";
    setTimeout(() => {
      copyBtn.textContent = "Copy text";
    }, 1100);
  } catch {
    copyBtn.textContent = "Copy failed";
    setTimeout(() => {
      copyBtn.textContent = "Copy text";
    }, 1100);
  }
});

downloadBtn.addEventListener("click", () => {
  if (!currentResult) return;
  const blob = new Blob([JSON.stringify(currentResult, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `extraction-${currentJobId || "result"}.json`;
  link.click();
  URL.revokeObjectURL(url);
});

tabButtons.forEach((button) => {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];

  if (!file) {
    statusText.textContent = "Choose a PDF before uploading.";
    setBadge(apiHealth, "Missing file", "pill-warm");
    return;
  }

  try {
    await uploadFile(file);
  } catch (error) {
    statusText.textContent = error.message || "Upload failed.";
    jobStatusEl.textContent = "failed";
    setBadge(apiHealth, "Upload error", "pill-danger");
    setProgress(0);
  }
});

resetResult();
fetchApiHealth();
