const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileMeta = document.getElementById("file-meta");
const clearBtn = document.getElementById("clear-btn");
const copyBtn = document.getElementById("copy-btn");
const statusText = document.getElementById("status-text");
const jobBadge = document.getElementById("job-badge");
const jobIdEl = document.getElementById("job-id");
const progressBar = document.getElementById("progress-bar");
const resultText = document.getElementById("result-text");
const metadataEl = document.getElementById("metadata");
const tablesEl = document.getElementById("tables");

let currentText = "";
let activePoll = null;

function setBadge(label, state = "muted") {
  jobBadge.className = `badge ${state}`;
  jobBadge.textContent = label;
}

function setProgress(percent) {
  progressBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
}

function resetOutput() {
  statusText.textContent = "Upload a file to start extraction.";
  jobIdEl.textContent = "";
  setBadge("Idle", "muted");
  setProgress(0);
  resultText.textContent = "Waiting for extraction result...";
  metadataEl.innerHTML = "";
  tablesEl.innerHTML = '<p class="meta">No tables yet.</p>';
  currentText = "";
}

function renderMetadata(metadata = {}) {
  const pairs = [
    ["Pages", metadata.pages ?? "-"],
    ["PDF Type", metadata.pdf_type ?? "-"],
    ["Confidence", metadata.confidence_score ?? "-"],
    ["Time", metadata.processing_time_seconds ?? "-"],
    ["Language", Array.isArray(metadata.languages_detected) ? metadata.languages_detected.join(", ") || "-" : "-"],
    ["OCR", metadata.ocr_engine ?? "-"],
  ];

  metadataEl.innerHTML = pairs
    .map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`)
    .join("");
}

function renderTables(tables = []) {
  if (!tables.length) {
    tablesEl.innerHTML = '<p class="meta">No tables extracted.</p>';
    return;
  }

  tablesEl.innerHTML = tables
    .map((table, index) => {
      const headers = table.headers ?? [];
      const rows = table.rows ?? [];
      const tableHead = headers.length
        ? `<tr>${headers.map((header) => `<th>${header || "&nbsp;"}</th>`).join("")}</tr>`
        : "";
      const bodyRows = rows
        .map(
          (row) =>
            `<tr>${row.map((cell) => `<td>${cell || "&nbsp;"}</td>`).join("")}</tr>`
        )
        .join("");

      return `
        <article class="table">
          <h3>Table ${index + 1} <span class="meta">(${table.extraction_method || "unknown"})</span></h3>
          <table>
            <thead>${tableHead}</thead>
            <tbody>${bodyRows || '<tr><td colspan="99" class="meta">No rows.</td></tr>'}</tbody>
          </table>
        </article>
      `;
    })
    .join("");
}

async function pollJob(jobId) {
  if (activePoll) {
    clearTimeout(activePoll);
  }

  const tick = async () => {
    try {
      const response = await fetch(`/api/v1/extract/${jobId}`);
      if (response.status === 202) {
        const body = await response.json();
        statusText.textContent = body.message || "Processing...";
        setBadge(body.status || "processing", "muted");
        setProgress(body.progress_percent || 0);
        activePoll = setTimeout(tick, 1200);
        return;
      }

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Request failed (${response.status})`);
      }

      const data = await response.json();
      currentText = data.text || "";
      resultText.textContent = currentText || "No text extracted.";
      statusText.textContent = "Extraction complete.";
      setBadge(data.status || "done", "ok");
      setProgress(100);
      renderMetadata(data.metadata);
      renderTables(data.tables || []);
      activePoll = null;
    } catch (error) {
      statusText.textContent = error.message || "Something went wrong.";
      setBadge("Error", "bad");
      setProgress(0);
      activePoll = null;
    }
  };

  tick();
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  fileMeta.textContent = file
    ? `${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MB`
    : "No file selected.";
});

clearBtn.addEventListener("click", () => {
  fileInput.value = "";
  if (activePoll) {
    clearTimeout(activePoll);
    activePoll = null;
  }
  resetOutput();
});

copyBtn.addEventListener("click", async () => {
  if (!currentText) return;
  try {
    await navigator.clipboard.writeText(currentText);
    copyBtn.textContent = "Copied";
    setTimeout(() => {
      copyBtn.textContent = "Copy text";
    }, 1200);
  } catch {
    copyBtn.textContent = "Copy failed";
    setTimeout(() => {
      copyBtn.textContent = "Copy text";
    }, 1200);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];

  if (!file) {
    statusText.textContent = "Choose a PDF before uploading.";
    setBadge("Missing file", "bad");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  setBadge("Uploading", "muted");
  statusText.textContent = "Uploading file...";
  setProgress(15);
  resultText.textContent = "Waiting for extraction result...";
  metadataEl.innerHTML = "";
  tablesEl.innerHTML = '<p class="meta">No tables yet.</p>';
  currentText = "";

  try {
    const response = await fetch("/api/v1/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Upload failed.");
    }

    jobIdEl.textContent = `Job ID: ${data.job_id}`;
    statusText.textContent = "File uploaded. Starting extraction...";
    setBadge("Queued", "muted");
    setProgress(30);
    pollJob(data.job_id);
  } catch (error) {
    statusText.textContent = error.message || "Upload failed.";
    setBadge("Error", "bad");
    setProgress(0);
  }
});

resetOutput();
