import { api, byId, closePeer, negotiate, newPeer, notice, setStatus, setupPairing } from "./common.js";

const video = byId("feed-video");
const timeline = byId("timeline");
const pageSize = 8;
let paired = false;
let cameraId = null;
let viewer = null;
let offset = 0;
let nextOffset = null;
let selection = null;
let lastPage = "";
let statusBusy = false;
let archiveRequest = 0;

const sampleLabels = {
  saved: "Saved", analysis_failed: "Inference unavailable", too_blurry: "Too blurry",
  duplicate: "Duplicate", min_interval: "Minimum interval", processing_error: "Processing error",
};
const dateTime = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "medium" });
const shortTime = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit", second: "2-digit" });

function stopViewer() {
  const old = viewer;
  viewer = null;
  closePeer(old);
  video.srcObject = null;
  video.controls = false;
  byId("feed-empty").hidden = false;
}

async function viewCamera(id) {
  if (viewer || typeof RTCPeerConnection === "undefined") return;
  const active = viewer = newPeer();
  byId("feed-message").textContent = "Connecting to camera";
  active.pc.addTransceiver("video", { direction: "recvonly" });
  active.pc.addEventListener("track", event => {
    if (viewer !== active) return;
    video.srcObject = event.streams[0] || new MediaStream([event.track]);
    video.controls = true;
    video.play().catch(() => {});
  });
  active.pc.addEventListener("connectionstatechange", () => {
    if (viewer === active && ["failed", "closed", "disconnected"].includes(active.pc.connectionState)) {
      stopViewer();
      byId("feed-message").textContent = "Video connection lost";
    }
  });
  try {
    await negotiate(active, "viewer", "Dashboard");
    if (id !== cameraId && viewer === active) stopViewer();
  } catch (error) {
    if (viewer === active) {
      stopViewer();
      byId("feed-message").textContent = error.message;
    }
  }
}

video.addEventListener("loadeddata", () => {
  if (viewer && video.srcObject) byId("feed-empty").hidden = true;
});

async function refreshStatus() {
  if (statusBusy) return;
  statusBusy = true;
  try {
    const status = await api("/live/status");
    const first = !paired;
    paired = true;
    byId("pair-form").hidden = true;
    notice();
    const camera = status.camera;
    const id = camera?.id ?? null;
    const live = camera?.state === "connected";
    if (cameraId !== id) {
      stopViewer();
      cameraId = id;
    }
    setStatus(live ? "Camera live" : camera ? "Connecting" : "Paired", live ? "live" : camera ? "pending" : "idle");
    byId("device-name").textContent = camera?.label || "No camera connected";
    byId("device-state").textContent = live ? "Streaming" : camera ? "Negotiating WebRTC" : "Waiting for a camera";
    byId("viewers").textContent = status.viewers;
    byId("sampled").textContent = camera?.sampled ?? 0;
    byId("stored").textContent = camera?.stored ?? 0;
    byId("sample-result").textContent = sampleLabels[camera?.last_result] || "None";
    if (live) void viewCamera(id);
    else byId("feed-message").textContent = camera ? "Waiting for video" : "No live feed";
    if (first) void loadArchive();
  } catch (error) {
    if (error.status === 401 || error.status === 403 || error.status === 503) {
      stopViewer();
      paired = false;
      cameraId = null;
      offset = 0;
      nextOffset = null;
      lastPage = "";
      archiveRequest += 1;
      renderArchive([]);
      byId("pair-form").hidden = false;
      setStatus("Not paired");
      byId("device-name").textContent = "No camera connected";
      byId("device-state").textContent = "Waiting for a camera";
      byId("feed-message").textContent = "No live feed";
      for (const id of ["viewers", "sampled", "stored"]) byId(id).textContent = "0";
      byId("sample-result").textContent = "None";
      if (error.status !== 401) notice(error.message);
    } else notice("Cannot reach the live server.");
  } finally {
    statusBusy = false;
  }
}

function showDetail(frame) {
  selection = frame?.id ?? null;
  byId("frame-detail").hidden = !frame;
  for (const button of timeline.children) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.id) === selection));
  }
  if (!frame) return;
  byId("frame-id").textContent = `#${frame.id}`;
  byId("frame-time").textContent = dateTime.format(new Date(frame.timestamp));
  byId("frame-size").textContent = `${frame.width} × ${frame.height}`;
  byId("frame-sharpness").textContent = `${frame.sharpness_score.toFixed(2)} · ${frame.quality}`;
  byId("frame-ocr").textContent = frame.ocr_text || "No text recognized";
  const list = byId("frame-objects");
  list.replaceChildren();
  for (const object of frame.objects) {
    const item = document.createElement("li");
    const confidence = document.createElement("strong");
    confidence.textContent = `${Math.round(object.confidence * 100)}%`;
    item.append(object.name, confidence);
    list.append(item);
  }
  if (!frame.objects.length) {
    const item = document.createElement("li");
    item.textContent = "No objects detected";
    list.append(item);
  }
}

function renderArchive(frames) {
  const focused = timeline.contains(document.activeElement) ? document.activeElement.dataset.id : null;
  timeline.replaceChildren();
  byId("archive-empty").hidden = Boolean(frames.length);
  byId("archive-empty").textContent = paired ? "No stored frames" : "No frames displayed";
  byId("newer").disabled = !offset;
  byId("older").disabled = nextOffset === null;
  for (const frame of frames) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "frame-tile";
    button.dataset.id = frame.id;
    const date = new Date(frame.timestamp);
    button.setAttribute("aria-label", `Frame ${frame.id}, ${dateTime.format(date)}`);
    const image = document.createElement("img");
    image.src = `/live/archive/${frame.id}/image`;
    image.alt = "";
    image.loading = "lazy";
    const time = document.createElement("time");
    time.dateTime = frame.timestamp;
    time.textContent = shortTime.format(date);
    button.append(image, time);
    button.addEventListener("click", () => showDetail(frame));
    timeline.append(button);
    if (button.dataset.id === focused) button.focus({ preventScroll: true });
  }
  showDetail(frames.find(frame => frame.id === selection) || frames[0]);
}

async function loadArchive() {
  if (!paired) return;
  const request = ++archiveRequest;
  try {
    const page = await api(`/frames?offset=${offset}&limit=${pageSize}`);
    if (request !== archiveRequest) return;
    nextOffset = page.next_offset;
    const key = JSON.stringify([offset, nextOffset, page.items.map(item => item.id)]);
    if (key !== lastPage) {
      lastPage = key;
      renderArchive(page.items);
    }
  } catch (error) {
    if (request === archiveRequest) notice(error.message);
  }
}

byId("newer").addEventListener("click", () => { offset = Math.max(0, offset - pageSize); void loadArchive(); });
byId("older").addEventListener("click", () => { if (nextOffset !== null) { offset = nextOffset; void loadArchive(); } });
setupPairing(refreshStatus);
window.addEventListener("pageshow", refreshStatus);
window.addEventListener("pagehide", stopViewer);
setInterval(refreshStatus, 1500);
setInterval(() => { if (!offset) void loadArchive(); }, 4000);
