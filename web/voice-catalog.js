export const locale = language => language === "fr" ? "fr-FR" : "en-US";

const voiceLocale = voice => String(voice.lang || "").replaceAll("_", "-");
const natural = /natural|neural|premium|enhanced/i;

export function chooseVoice(voices, language, allowBrowser) {
  const tag = locale(language).toLowerCase();
  const matching = voices.filter(voice => voiceLocale(voice).toLowerCase().split("-")[0] === language
    && (voice.localService || allowBrowser));
  const score = voice => Number(voiceLocale(voice).toLowerCase() === tag) * 4
    + Number(natural.test(voice.name)) * 2 + Number(Boolean(voice.localService));
  const voice = matching.sort((left, right) => score(right) - score(left))[0];
  return voice ? { voice, language: voiceLocale(voice), regional: voiceLocale(voice).toLowerCase() !== tag } : null;
}

export function voiceCatalog(engine) {
  let voices = [];
  const listeners = new Set();
  const refresh = () => {
    try { voices = Array.from(engine?.getVoices() ?? []); } catch { voices = []; }
    for (const notify of listeners) notify();
  };
  engine?.addEventListener?.("voiceschanged", refresh);
  refresh();
  const select = (language, allowBrowser) => chooseVoice(voices, language, allowBrowser);
  return {
    select,
    subscribe(notify) { listeners.add(notify); return () => listeners.delete(notify); },
    wait(language, allowBrowser, signal) {
      const choice = select(language, allowBrowser);
      if (choice || signal?.aborted || !engine) return Promise.resolve(choice);
      return new Promise(resolve => {
        const done = result => {
          clearTimeout(timer);
          listeners.delete(changed);
          signal?.removeEventListener("abort", abort);
          resolve(result);
        };
        const abort = () => done(null);
        const changed = () => {
          const result = select(language, allowBrowser);
          if (result) done(result);
        };
        const timer = setTimeout(() => done(select(language, allowBrowser)), 800);
        listeners.add(changed);
        signal?.addEventListener("abort", abort, { once: true });
        refresh();
      });
    },
  };
}
