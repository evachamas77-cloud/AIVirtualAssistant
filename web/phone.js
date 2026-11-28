import { suspendManualVoice } from "./assistant.js";
import { api, byId, closePeer, negotiate, newPeer, notice, setStatus, setupPairing } from "./common.js";
import { answerError, questionSource } from "./mobile-question.js";
import { sceneMessage, strings } from "./mobile-strings.js";
import { PhoneCamera } from "./phone-camera.js";
import { SpeechRecognition, localAvailability, locale, mobileSpeaker, voices } from "./voice.js";
import { State, VoiceSession } from "./voice-session.js";

const language = byId("mobile-language");
const consent = byId("session-consent");
const browserSpeech = byId("speech-consent");
const preview = byId("preview");
const pairing = byId("pair-form");
const pack = byId("mobile-speech-pack");
const manual = byId("manual-tools");
const start = byId("start-assistant");
const stop = byId("stop-assistant");
const talk = byId("talk-again");
const typed = byId("mobile-question");
const describeChanges = byId("describe-changes");
const text = key => strings[language.value]?.[key] ?? strings.en[key];
let paired = false, pairAttempt = 0, speechPack = "unavailable", packCheck = 0;
let display = { state: State.IDLE, note: "idle", active: false, hasAnswer: false };
let scenePoll = null, lastChange = null;

const camera = new PhoneCamera({
  getMedia: navigator.mediaDevices?.getUserMedia?.bind(navigator.mediaDevices),
  createPeer: typeof RTCPeerConnection === "function" ? newPeer : null,
  negotiate, closePeer, changed: updateCamera, lost: loseConnection,
  label: /Android/.test(navigator.userAgent) ? "Android phone" : "Browser camera",
});

function showAnswer(answer = "") {
  byId("mobile-answer").textContent = answer;
  byId("mobile-answer-section").hidden = !answer;
}

function announce(message) {
  byId("voice-announcement").textContent = message;
}

function setVoiceDisplay(next) {
  const wasActive = display.active;
  const restoreFocus = document.activeElement === stop && !document.hidden;
  display = next;
  if (next.state === State.CONNECTING && !wasActive) {
    announce("");
    showAnswer();
    manual.open = false;
  } else if (wasActive && !next.active) {
    scenePoll?.abort();
    lastChange = null;
    consent.checked = false;
    typed.value = "";
    byId("type-tools").open = false;
    showAnswer();
  }
  if (next.note === "pairRequired") paired = false;
  render();
  if (next.note === "pairRequired") byId("pair-code").focus();
  else if (wasActive && !next.active && restoreFocus) start.focus();
}

function render() {
  const { state, note, active, hasAnswer } = display;
  const stateLabel = byId("voice-state");
  stateLabel.dataset.state = state;
  stateLabel.textContent = text(note);
  start.disabled = active || Boolean(camera.pending) || (camera.peer && camera.stage === "connecting");
  stop.disabled = !active;
  const canAsk = active && [State.LISTENING, State.READY, State.ERROR].includes(state);
  talk.disabled = (!canAsk && note !== "announcing") || state === State.LISTENING || !SpeechRecognition ||
    (!browserSpeech.checked && speechPack !== "available");
  byId("repeat-answer").disabled = !canAsk || !hasAnswer;
  byId("stop-speaking").hidden = state !== State.SPEAKING;
  byId("mobile-ask").disabled = !active || state === State.CONNECTING;
  byId("type-tools").hidden = !active;
  manual.hidden = active;
  browserSpeech.disabled = pack.disabled = active;
  const selectedVoice = voices.select(language.value, browserSpeech.checked);
  byId("mobile-voice-choice").textContent = !selectedVoice ? text("voiceMissing") : selectedVoice.regional
    ? `${text("regionalVoice")} (${selectedVoice.language})` : "";
  pairing.hidden = paired;
  const cameraLabel = camera.stage === "cameraOff" ? (paired ? "paired" : "unpaired") : camera.stage;
  const cameraState = ["cameraReady", "streaming"].includes(camera.stage) ? "live" :
    camera.stage === "cameraOff" ? "idle" : "pending";
  setStatus(text(cameraLabel), cameraState);
  byId("start-camera").disabled = Boolean(camera.stream || camera.pending);
  byId("start-stream").disabled = !camera.stream || !paired || Boolean(camera.peer);
  byId("stop").disabled = !camera.stream && !camera.pending && !camera.peer;
}

function updateCamera() {
  if (preview.srcObject !== camera.stream) {
    preview.srcObject = camera.stream;
    if (camera.stream) preview.play().catch(() => {});
  }
  byId("preview-empty").hidden = Boolean(camera.stream);
  render();
}

function loseConnection(reason) {
  if (display.active) voice.stop(reason);
  else {
    camera.close();
    notice(text(reason));
  }
}

async function checkPairing(signal) {
  const attempt = ++pairAttempt;
  const timeout = AbortSignal.timeout?.(7000);
  const requestSignal = signal && timeout && AbortSignal.any ? AbortSignal.any([signal, timeout]) : signal || timeout;
  function setPaired(value) {
    if (!signal?.aborted && attempt === pairAttempt) {
      paired = value;
      render();
    }
  }
  try {
    await api("/live/status", { signal: requestSignal });
    setPaired(true);
  } catch (error) {
    signal?.throwIfAborted();
    const reason = error.status === 401 ? "pairRequired" : error.status === 403 ? "badHttps" : "connectionLost";
    if (error.status === 401 || error.status === 403) setPaired(false);
    throw Object.assign(error, { reason });
  }
}

async function refreshPairing() {
  try {
    await checkPairing();
    if ([text("connectionLost"), text("offline")].includes(byId("notice").textContent)) notice();
  } catch (error) {
    if (error.reason === "pairRequired") {
      if (!paired && (display.active || camera.peer)) loseConnection("pairRequired");
    } else if (!display.active && !camera.stream && display.state === State.IDLE) {
      const warning = text(navigator.onLine ? error.reason : "offline");
      if (byId("notice").textContent !== warning) notice(warning);
    }
  }
}

const voice = new VoiceSession({
  async connect(signal) {
    if (!window.isSecureContext) throw { reason: "badHttps" };
    if (!navigator.onLine) throw { reason: "offline" };
    await checkPairing(signal);
    await camera.open(signal);
    await camera.streamToServer(signal);
  },
  disconnect: () => camera.close(),
  recognition(settings) {
    const local = speechPack === "available";
    if (!SpeechRecognition || (!local && !settings.allowBrowser)) return null;
    const recognition = new SpeechRecognition();
    if ("processLocally" in recognition) recognition.processLocally = local;
    recognition.lang = locale(settings.language);
    recognition.interimResults = recognition.continuous = false;
    return recognition;
  },
  speaker: mobileSpeaker(),
  async ask(question, settings, signal) {
    const source = questionSource(question);
    try {
      return await api("/ai/ask", {
        method: "POST", headers: { "Content-Type": "application/json" }, signal,
        body: JSON.stringify({ question, language: settings.language, consent: settings.consent, ...source }),
      });
    } catch (error) {
      throw Object.assign(error, { reason: answerError(error.status, source.scope) });
    }
  },
  update: setVoiceDisplay, announce, answer: showAnswer, text,
  report(timings) {
    byId("mobile-timings").textContent = Object.entries(timings).filter(([, value]) => Number.isFinite(value))
      .map(([stage, value]) => `${stage}: ${Math.round(value)} ms`).join(" | ");
  },
});

async function checkSpeechPack() {
  const attempt = ++packCheck;
  const result = await localAvailability(language.value);
  if (attempt !== packCheck) return;
  speechPack = result;
  pack.hidden = !(speechPack === "downloadable" && typeof SpeechRecognition?.install === "function");
  render();
}

function localize() {
  document.documentElement.lang = language.value;
  document.querySelectorAll("[data-label]").forEach(node => { node.textContent = text(node.dataset.label); });
  render();
}

try { if (localStorage.getItem("visual-assistant-language") === "fr") language.value = "fr"; } catch {}
language.addEventListener("change", () => {
  try { localStorage.setItem("visual-assistant-language", language.value); } catch {}
  speechPack = "unavailable";
  notice();
  localize();
  voice.setLanguage(language.value);
  void checkSpeechPack().then(() => {
    if (speechPack === "available" && voice.state === State.ERROR && display.note === "micUnavailable") voice.talkAgain();
  });
});
pack.addEventListener("click", async () => {
  pack.disabled = true;
  try {
    const installed = await SpeechRecognition.install({ langs: [locale(language.value)], processLocally: true });
    notice(text(installed ? "packReady" : "packFailed"));
    await checkSpeechPack();
  } catch {
    notice(text("packFailed"));
  } finally {
    pack.disabled = display.active;
  }
});

setupPairing(async () => {
  await checkPairing();
  if (!display.active) {
    announce(text("paired"));
    (consent.checked ? start : consent).focus();
  }
}, {
  describe: error => text(error.status === 401 || error.status === 403 ? "pairFailed" : "connectionLost"),
  insecure: () => text("badHttps"),
});

start.addEventListener("click", () => {
  notice();
  if (!consent.checked) {
    void voice.start({ consent: false });
    consent.focus();
    return;
  }
  if (!suspendManualVoice()) return notice(text("busy"));
  void voice.start({ consent: true, language: language.value, allowBrowser: browserSpeech.checked });
  if (display.active) stop.focus();
});
stop.addEventListener("click", () => voice.stop());
talk.addEventListener("click", () => voice.talkAgain());
byId("repeat-answer").addEventListener("click", () => voice.repeat());
byId("stop-speaking").addEventListener("click", () => voice.stopSpeaking());
browserSpeech.addEventListener("change", render);
describeChanges.addEventListener("change", () => {
  if (!describeChanges.checked) {
    scenePoll?.abort();
    voice.cancelNotice();
  }
});

async function pollChanges() {
  if (scenePoll || !display.active || !describeChanges.checked || document.hidden) return;
  const poll = scenePoll = new AbortController();
  try {
    const { change } = await api("/live/changes", { signal: poll.signal });
    if (!change || poll.signal.aborted || change.id === lastChange || !describeChanges.checked) return;
    lastChange = change.id;
    const message = sceneMessage(change, language.value);
    if (message) voice.offerNotice(message);
  } catch { /* Optional local hints must not disrupt a conversation. */ }
  finally { if (scenePoll === poll) scenePoll = null; }
}
setInterval(pollChanges, 1800);
consent.addEventListener("change", () => { if (!consent.checked && display.active) voice.stop(); });
byId("mobile-question-form").addEventListener("submit", event => {
  event.preventDefault();
  if (voice.askTyped(typed.value)) typed.value = "";
});

byId("start-camera").addEventListener("click", async () => {
  notice();
  try { await camera.open(); } catch (error) {
    if (error.name !== "AbortError") notice(text(error.reason || "cameraUnavailable"));
  }
});
byId("start-stream").addEventListener("click", async () => {
  notice();
  try { await camera.streamToServer(); } catch (error) {
    if (error.name === "AbortError") return;
    if (error.reason === "pairRequired") {
      paired = false;
      render();
      byId("pair-code").focus();
    }
    notice(text(error.reason || "connectionLost"));
  }
});
byId("stop").addEventListener("click", () => { camera.close(); notice(); });

window.addEventListener("offline", () => loseConnection("offline"));
window.addEventListener("online", () => { notice(); void refreshPairing(); });
window.addEventListener("pagehide", () => {
  if (display.active) voice.stop("paused");
  else camera.close();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && (display.active || camera.stream || camera.pending)) loseConnection("paused");
});
window.addEventListener("pageshow", refreshPairing);
setInterval(() => { if (!document.hidden) void refreshPairing(); }, 7000);
localize();
voices.subscribe(render);
void checkSpeechPack();
void refreshPairing();

const install = byId("install");
let installEvent;
window.addEventListener("beforeinstallprompt", event => {
  event.preventDefault();
  installEvent = event;
  install.hidden = false;
});
install.addEventListener("click", async () => {
  if (!installEvent) return;
  try { await installEvent.prompt(); } finally {
    installEvent = null;
    install.hidden = true;
  }
});
if (window.isSecureContext && "serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
