import assert from "node:assert/strict";
import test from "node:test";
import { chooseVoice, voiceCatalog } from "../web/voice-catalog.js";

const english = { lang: "en-US", name: "English", localService: true };
const frCanada = { lang: "fr-CA", name: "French Canada", localService: true };
const frFrance = { lang: "fr-FR", name: "French France", localService: true };
const frNatural = { ...frFrance, name: "French Natural" };

test("voice selection is explicitly same-language, locale-aware and consent-aware", () => {
  const pool = [english, frCanada, frFrance, frNatural];
  assert.equal(chooseVoice(pool, "en", false).voice, english);
  assert.equal(chooseVoice(pool, "fr", false).voice, frNatural);
  assert.deepEqual(chooseVoice([english, frCanada], "fr", false), { voice: frCanada, language: "fr-CA", regional: true });
  assert.equal(chooseVoice([english], "fr", true), null);
  assert.equal(chooseVoice([{ ...frFrance, localService: false }], "fr", false), null);
});

test("a request waits for arrival of a compatible voice via voiceschanged", async () => {
  const engine = Object.assign(new EventTarget(), { current: [english], getVoices() { return this.current; } });
  const catalog = voiceCatalog(engine);
  const pending = catalog.wait("fr", false);
  engine.current = [english, frFrance];
  engine.dispatchEvent(new Event("voiceschanged"));
  assert.equal((await pending).voice, frFrance);
});

test("browser synthesis never starts a French utterance on an English-only device", async () => {
  const engine = Object.assign(new EventTarget(), { getVoices: () => [english], speak() { throw new Error("Unexpected speech"); } });
  globalThis.window = { speechSynthesis: engine, SpeechSynthesisUtterance: class {} };
  const { mobileSpeaker } = await import("../web/voice.js");
  const speaker = mobileSpeaker();
  assert.equal(await speaker.play("Bonjour", { language: "fr", allowBrowser: true }), "voiceMissing");
});
