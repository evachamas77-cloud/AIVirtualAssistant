import { api, byId } from "./common.js";
import { wireVoice } from "./voice.js";

const icon = name => '<svg aria-hidden="true"><use href="/ui/icons.svg#' + name + '"/></svg>';
const root = byId("assistant");
root.innerHTML = `
  <header class="section-heading assistant-heading">
    <h2 id="assistant-heading">${icon("sparkles")} Visual assistant</h2>
    <div class="assistant-tabs" role="tablist" aria-label="Assistant views">
      <button id="ai-ask-tab" role="tab" aria-selected="true" aria-controls="ai-ask-panel" type="button">Ask AI</button>
      <button id="ai-search-tab" role="tab" aria-selected="false" aria-controls="ai-search-panel" tabindex="-1" type="button">Semantic search</button>
    </div>
  </header>
  <div id="ai-ask-panel" role="tabpanel" aria-labelledby="ai-ask-tab">
    <div class="assistant-layout">
      <form id="ai-ask-form">
        <div class="assistant-options">
          <label>Scene <select id="ai-scope">
            <option value="auto">Auto select</option><option value="current">Current camera</option>
            <option value="history">Archive search</option><option value="frame" hidden>Selected frame</option>
          </select></label>
          <label>Answer language <select id="ai-language"><option value="en">English</option><option value="fr">Français</option></select></label>
        </div>
        <p id="ai-selected" class="assistant-subtle" hidden></p>
        <label for="ai-question">Question</label>
        <textarea id="ai-question" rows="3" maxlength="800" required placeholder="What can you see in front of me?"></textarea>
        <div class="assistant-actions">
          <button type="button" id="ai-mic" class="icon-button" title="Start listening" aria-label="Start listening">${icon("mic")}</button>
          <button type="button" id="ai-stop-mic" class="icon-button" title="Stop listening" aria-label="Stop listening" disabled>${icon("stop")}</button>
          <button type="button" id="ai-install-speech" hidden>Download speech pack</button>
          <button type="submit" id="ai-ask" class="primary">${icon("sparkles")} Ask</button>
        </div>
        <p id="ai-voice-status" class="assistant-subtle" role="status"></p>
        <label class="assistant-check"><input type="checkbox" id="ai-browser-speech"> Allow browser-managed speech, which may send audio or answer text to the browser provider.</label>
        <label class="assistant-check"><input type="checkbox" id="ai-consent" required> Send my question and selected images to OpenAI for this answer.</label>
        <p class="assistant-subtle">Only pressing Ask sends images. This uses the project's API budget.</p>
      </form>
      <section class="assistant-answer" aria-labelledby="ai-answer-heading">
        <div class="section-heading"><h3 id="ai-answer-heading">Answer</h3><span id="ai-source" class="transport"></span></div>
        <p id="ai-ask-status" class="assistant-message" role="status">No answer yet.</p>
        <progress id="ai-ask-progress" aria-label="Analyzing images" hidden></progress>
        <p id="ai-answer" class="ocr-text" aria-live="polite"></p>
        <div class="assistant-actions">
          <button type="button" id="ai-play" disabled>${icon("speaker")} Speak</button>
          <button type="button" id="ai-replay" disabled>Replay</button>
          <button type="button" id="ai-stop-audio" class="icon-button" title="Stop speaking" aria-label="Stop speaking" disabled>${icon("stop")}</button>
        </div>
        <label class="assistant-check"><input type="checkbox" id="ai-auto-speak"> Speak answers automatically</label>
        <div id="ai-evidence" class="assistant-results" aria-label="Supporting frames"></div>
      </section>
    </div>
  </div>
  <div id="ai-search-panel" role="tabpanel" aria-labelledby="ai-search-tab" hidden>
    <form id="ai-search-form">
      <label for="ai-query">Search phrase</label>
      <div class="input-row"><input type="search" id="ai-query" maxlength="800" required placeholder="A television in a living room">
        <button type="submit" class="primary" id="ai-search">${icon("search")} Search</button></div>
      <div class="assistant-options search-options">
        <label>From <input id="ai-start" type="date"></label>
        <label>Through <input id="ai-end" type="date"></label>
      </div>
    </form>
    <div class="index-line"><p id="ai-index" class="assistant-subtle" role="status"></p>
      <button type="button" id="ai-backfill">Index archive</button></div>
    <p id="ai-search-status" class="assistant-message" role="status">No search yet.</p>
    <progress id="ai-search-progress" aria-label="Searching archive" hidden></progress>
    <div id="ai-results" class="assistant-results" aria-label="Ranked archive matches"></div>
  </div>`;

const get = id => byId("ai-" + id);
const askTab = get("ask-tab");
const searchTab = get("search-tab");
const clock = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "medium" });
const sourceLabels = {
  live: "Live camera", history: "Archive matches", frame: "Selected archive frame",
  archive_fallback: "Latest archive frame (not live)",
};
const voice = wireVoice(get, message => { get("voice-status").textContent = message; });
let selectedFrame = null;
let askBusy = false;
let searchBusy = false;

export function suspendManualVoice() {
  if (askBusy) return false;
  voice.cancelListening();
  voice.stopSpeech();
  return true;
}

function showTab(search, focus = false) {
  for (const [tab, visible] of [[askTab, !search], [searchTab, search]]) {
    tab.setAttribute("aria-selected", String(visible));
    tab.tabIndex = visible ? 0 : -1;
    byId(tab.getAttribute("aria-controls")).hidden = !visible;
    if (visible && focus) tab.focus();
  }
  if (search) {
    voice.cancelListening();
    void indexStatus();
  }
}
askTab.addEventListener("click", () => showTab(false));
searchTab.addEventListener("click", () => showTab(true));
for (const tab of [askTab, searchTab]) {
  tab.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    showTab(event.key === "End" || (event.key !== "Home" && tab === askTab), true);
  });
}

function feedback(target, message, error = false) {
  target.textContent = message;
  target.classList.toggle("error", error);
}

function message(error) {
  return error.status === 401 ? "Pair this device first, then try again." : error.message;
}

function showIndex(index) {
  const parts = [];
  if (!index.available) parts.push("Local model not configured");
  parts.push(`${index.indexed} indexed · ${index.pending} pending`);
  if (index.unreadable) parts.push(`${index.unreadable} unreadable`);
  if (index.error) parts.push(index.error);
  get("index").textContent = parts.join(" · ");
  get("backfill").disabled = !index.available;
}

async function indexStatus() {
  try {
    showIndex(await api("/ai/index"));
  } catch (error) {
    get("index").textContent = message(error);
    get("backfill").disabled = true;
  }
}

function selectFrame(item) {
  selectedFrame = item.frame_id;
  get("scope").querySelector('[value="frame"]').hidden = false;
  get("scope").value = "frame";
  get("selected").textContent = `Frame #${item.frame_id} · ${clock.format(new Date(item.timestamp))}`;
  get("selected").hidden = false;
  showTab(false);
  get("question").focus();
}
get("scope").addEventListener("change", () => { get("selected").hidden = get("scope").value !== "frame"; });

function cards(container, items, ranked) {
  container.replaceChildren();
  for (const item of items) {
    if (!/^\/(?:live\/archive\/\d+\/image|ai\/evidence\/live-[\w-]+)$/.test(item.image_url)) continue;
    const tile = document.createElement("article");
    tile.className = "evidence-item";
    const image = document.createElement("img");
    image.src = item.image_url;
    image.alt = item.source === "live" ? "Live snapshot used for the answer" : `Archived frame ${item.frame_id}`;
    image.loading = "lazy";
    const time = document.createElement("time");
    time.dateTime = item.timestamp;
    time.textContent = clock.format(new Date(item.timestamp));
    const name = document.createElement("strong");
    name.textContent = item.source === "live" ? "Live snapshot" : `Frame #${item.frame_id}`;
    tile.append(image, name, time);
    if (ranked) {
      const score = document.createElement("span");
      score.textContent = `Similarity ${Number(item.similarity).toFixed(3)}`;
      const use = document.createElement("button");
      use.type = "button";
      use.textContent = "Ask about frame";
      use.addEventListener("click", () => selectFrame(item));
      tile.append(score, use);
    }
    container.append(tile);
  }
}

get("ask-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (askBusy) return;
  const question = get("question").value.trim();
  if (!question || !get("consent").checked) {
    feedback(get("ask-status"), "Enter a question and confirm image sharing.", true);
    return;
  }
  const language = get("language").value;
  const scope = get("scope").value;
  const body = { question, language, scope, consent: true };
  if (scope === "frame") body.frame_id = selectedFrame;
  if (scope === "history") {
    if (get("start").value) body.start = get("start").value;
    if (get("end").value) body.end = get("end").value;
  }
  askBusy = true;
  get("ask").disabled = true;
  get("ask-progress").hidden = false;
  get("consent").checked = false;
  voice.cancelListening();
  voice.reset();
  get("source").textContent = "";
  get("answer").textContent = "";
  get("evidence").replaceChildren();
  feedback(get("ask-status"), "Analyzing selected images...");
  try {
    const result = await api("/ai/ask", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    get("source").textContent = sourceLabels[result.source] || "";
    get("answer").lang = language;
    get("answer").textContent = result.answer;
    feedback(get("ask-status"), "");
    cards(get("evidence"), result.evidence, false);
    voice.setAnswer(result.answer, language);
  } catch (error) {
    feedback(get("ask-status"), message(error), true);
  } finally {
    askBusy = false;
    get("ask").disabled = false;
    get("ask-progress").hidden = true;
  }
});

get("search-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (searchBusy) return;
  const query = get("query").value.trim();
  if (!query) return feedback(get("search-status"), "Enter a phrase to search.", true);
  const params = new URLSearchParams({ q: query });
  for (const name of ["start", "end"]) if (get(name).value) params.set(name, get(name).value);
  searchBusy = true;
  get("search").disabled = true;
  get("search-progress").hidden = false;
  get("results").replaceChildren();
  feedback(get("search-status"), "Searching the local archive...");
  try {
    const result = await api("/ai/semantic-search?" + params);
    showIndex(result.index);
    cards(get("results"), result.items, true);
    const count = result.items.length;
    feedback(get("search-status"), count ? `${count} ranked ${count === 1 ? "frame" : "frames"}`
      : result.index.pending ? "No indexed frames for this search yet. Backfill may still be running."
      : "No indexed frames in this date range.");
  } catch (error) {
    feedback(get("search-status"), message(error), true);
  } finally {
    searchBusy = false;
    get("search").disabled = false;
    get("search-progress").hidden = true;
  }
});

get("backfill").addEventListener("click", async () => {
  get("backfill").disabled = true;
  try {
    const index = await api("/ai/index/backfill", { method: "POST" });
    showIndex(index);
    feedback(get("search-status"), index.pending ? "Archive indexing is running." : "Archive is indexed.");
  } catch (error) {
    feedback(get("search-status"), message(error), true);
    get("backfill").disabled = false;
  }
});
setInterval(() => { if (!get("search-panel").hidden) void indexStatus(); }, 6000);
