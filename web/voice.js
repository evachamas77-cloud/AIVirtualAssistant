export const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const speakEngine = window.speechSynthesis;
export const locale = language => language === "fr" ? "fr-FR" : "en-US";

export async function localAvailability(language) {
  if (typeof SpeechRecognition?.available !== "function") return "unavailable";
  try {
    return await SpeechRecognition.available({ langs: [locale(language)], processLocally: true });
  } catch {
    return "unavailable";
  }
}

function preferredVoice(language, allowBrowser) {
  const voices = speakEngine.getVoices().filter(voice => voice.lang.toLowerCase().startsWith(language));
  return voices.find(voice => voice.localService && voice.lang === locale(language))
    || voices.find(voice => voice.localService) || (allowBrowser && voices[0]);
}

export function mobileSpeaker() {
  let finish = null;
  return {
    cancel() {
      finish?.("cancelled");
      speakEngine?.cancel();
    },
    play(text, { language, allowBrowser }) {
      if (!speakEngine || !window.SpeechSynthesisUtterance || finish) return Promise.resolve("unavailable");
      const voice = preferredVoice(language, allowBrowser);
      if (!voice && !allowBrowser) return Promise.resolve("unavailable");
      return new Promise(resolve => {
        const speech = new SpeechSynthesisUtterance(text);
        speech.lang = locale(language);
        if (voice) speech.voice = voice;
        let timer;
        const done = result => {
          if (finish !== done) return;
          finish = null;
          clearTimeout(timer);
          speech.onstart = speech.onend = speech.onerror = null;
          resolve(result);
        };
        finish = done;
        speech.onstart = () => {
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
  let recognition, utterance, answer;
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
    utterance = null;
    speakEngine?.cancel();
    silence.disabled = true;
  }

  function speak() {
    if (!speakEngine || !answer) return;
    stopSpeech();
    const voice = preferredVoice(answer.language, browserConsent.checked);
    if (!voice && !browserConsent.checked) {
      announce("No matching local voice is available. Keep reading or allow browser-managed speech.");
      return;
    }
    const speech = utterance = new SpeechSynthesisUtterance(answer.text);
    speech.lang = locale(answer.language);
    if (voice) speech.voice = voice;
    speech.onstart = () => { if (utterance === speech) announce("Speaking answer."); };
    speech.onend = () => { if (utterance === speech) { stopSpeech(); announce("Playback finished."); } };
    speech.onerror = () => { if (utterance === speech) { stopSpeech(); announce("Speech playback is unavailable."); } };
    silence.disabled = false;
    speakEngine.speak(speech);
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
