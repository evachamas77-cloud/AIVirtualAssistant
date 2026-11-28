import { locale, voiceCatalog } from "./voice-catalog.js";

export const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const speakEngine = window.speechSynthesis;
export { locale };
export const voices = voiceCatalog(speakEngine);

export async function localAvailability(language) {
  if (typeof SpeechRecognition?.available !== "function") return "unavailable";
  try {
    return await SpeechRecognition.available({ langs: [locale(language)], processLocally: true });
  } catch {
    return "unavailable";
  }
}

export function mobileSpeaker() {
  let attempt = null;
  return {
    cancel() {
      if (attempt) {
        attempt.abort();
        attempt = null;
        speakEngine?.cancel();
      }
    },
    async play(text, { language, allowBrowser }, onStart = () => {}) {
      if (!speakEngine || !window.SpeechSynthesisUtterance || attempt) return "unavailable";
      const current = attempt = new AbortController();
      const choice = voices.select(language, allowBrowser) || await voices.wait(language, allowBrowser, current.signal);
      if (current.signal.aborted || !choice) {
        if (attempt === current) attempt = null;
        return current.signal.aborted ? "cancelled" : "voiceMissing";
      }
      return await new Promise(resolve => {
        const speech = new SpeechSynthesisUtterance(text);
        speech.lang = choice.language;
        speech.voice = choice.voice;
        let timer;
        const done = result => {
          if (attempt !== current) return;
          attempt = null;
          clearTimeout(timer);
          current.signal.removeEventListener("abort", aborted);
          speech.onstart = speech.onend = speech.onerror = null;
          resolve(result);
        };
        const aborted = () => done("cancelled");
        current.signal.addEventListener("abort", aborted, { once: true });
        speech.onstart = () => {
          onStart();
          clearTimeout(timer);
          timer = setTimeout(expired, 90000);
        };
        speech.onend = () => done("played");
        speech.onerror = () => done("unavailable");
        const expired = () => { done("unavailable"); speakEngine.cancel(); };
        timer = setTimeout(expired, 5000);
        try { speakEngine.speak(speech); } catch { done("unavailable"); }
      });
    },
  };
}

export function wireVoice(get, announce) {
  const question = get("question");
  const lang = get("language");
  const browserConsent = get("browser-speech");
  const mic = get("mic");
  const stop = get("stop-mic");
  const pack = get("install-speech");
  const play = get("play");
  const replay = get("replay");
  const silence = get("stop-audio");
  let recognition, answer;
  const playback = mobileSpeaker();
  let playAttempt = 0;
  let opening = null;

  function setListening(active) {
    mic.disabled = !SpeechRecognition || active;
    stop.disabled = !active;
  }

  function stopListening() {
    opening = null;
    if (recognition) {
      try { recognition.stop(); } catch {}
    } else setListening(false);
  }

  function cancelListening() {
    opening = null;
    const session = recognition;
    recognition = null;
    if (session) {
      try { session.abort(); } catch {}
    }
    setListening(false);
  }

  mic.addEventListener("click", async () => {
    if (!SpeechRecognition || recognition || opening) return;
    stopSpeech();
    const attempt = opening = {};
    pack.hidden = true;
    setListening(true);
    announce("Opening microphone...");
    const language = lang.value;
    const availability = await localAvailability(language);
    if (opening !== attempt) return;
    opening = null;
    const local = availability === "available";
    if (!local && !browserConsent.checked) {
      setListening(false);
      pack.hidden = !(availability === "downloadable" && typeof SpeechRecognition.install === "function");
      announce("On-device recognition is unavailable. Type instead or allow browser-managed speech.");
      return;
    }
    const session = recognition = new SpeechRecognition();
    if ("processLocally" in session) session.processLocally = local;
    session.lang = locale(language);
    session.interimResults = false;
    session.continuous = false;
    session.onstart = () => announce("Listening...");
    session.onresult = event => {
      if (recognition !== session) return;
      const transcript = Array.from(event.results).slice(event.resultIndex).map(result => result[0].transcript).join(" ");
      question.value = [question.value.trim(), transcript.trim()].filter(Boolean).join(" ").slice(0, 800);
      announce("Transcript ready. Review it before asking.");
    };
    session.onerror = event => {
      if (recognition !== session || event.error === "aborted") return;
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        announce("Microphone or speech service permission was denied. You can type your question.");
      } else if (event.error === "no-speech") {
        announce("No speech detected. Try again or type your question.");
      } else {
        announce("Speech recognition could not complete. Type your question instead.");
      }
    };
    session.onend = () => {
      if (recognition === session) {
        recognition = null;
        setListening(false);
        if (get("voice-status").textContent === "Listening...") announce("Microphone off.");
      }
    };
    try {
      session.start();
    } catch {
      recognition = null;
      setListening(false);
      announce("Could not start speech recognition. Type your question instead.");
    }
  });
  stop.addEventListener("click", stopListening);
  lang.addEventListener("change", () => {
    cancelListening();
    stopSpeech();
    pack.hidden = true;
  });
  pack.addEventListener("click", async () => {
    pack.disabled = true;
    try {
      const installed = await SpeechRecognition.install({ langs: [locale(lang.value)], processLocally: true });
      announce(installed ? "Speech pack installed. The microphone is ready." : "Speech pack unavailable. Type instead.");
      pack.hidden = installed;
    } catch {
      announce("The speech pack could not be installed. Type instead.");
    } finally {
      pack.disabled = false;
    }
  });

  function stopSpeech() {
    ++playAttempt;
    playback.cancel();
    silence.disabled = true;
  }

  async function speak() {
    if (!speakEngine || !answer) return;
    if (recognition || opening) return announce(lang.value === "fr"
      ? "Arr\u00eatez le microphone avant d'\u00e9couter la r\u00e9ponse." : "Stop the microphone before playing the answer.");
    stopSpeech();
    const selected = answer;
    if (selected.language !== lang.value) return;
    const current = playAttempt;
    silence.disabled = false;
    const result = await playback.play(selected.text, { language: selected.language, allowBrowser: browserConsent.checked },
      () => { if (current === playAttempt) announce("Speaking answer."); });
    if (current !== playAttempt) return;
    silence.disabled = true;
    if (result === "voiceMissing") announce(selected.language === "fr"
      ? "Aucune voix fran\u00e7aise compatible. Installez une voix fran\u00e7aise dans les param\u00e8tres de l'appareil."
      : "No compatible English voice. Install an English voice in your device's text-to-speech settings.");
    else if (result !== "cancelled") announce(result === "played" ? "Playback finished." : "Speech playback is unavailable.");
  }
  for (const button of [play, replay]) button.addEventListener("click", speak);
  silence.addEventListener("click", () => { stopSpeech(); announce("Playback stopped."); });
  browserConsent.addEventListener("change", () => {
    if (!browserConsent.checked) { cancelListening(); stopSpeech(); }
  });
  window.addEventListener("pagehide", () => { cancelListening(); stopSpeech(); });
  document.addEventListener("visibilitychange", () => { if (document.hidden) cancelListening(); });
  setListening(false);
  if (!SpeechRecognition) announce("Speech recognition is unavailable in this browser. You can type.");

  return {
    cancelListening,
    stopSpeech,
    reset() {
      answer = null;
      stopSpeech();
      play.disabled = replay.disabled = true;
    },
    setAnswer(text, language) {
      answer = { text, language };
      play.disabled = replay.disabled = !speakEngine;
      if (get("auto-speak").checked) speak();
    },
  };
}
