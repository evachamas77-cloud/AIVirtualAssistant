import assert from "node:assert/strict";
import test from "node:test";
import { State, VoiceSession } from "../web/voice-session.js";

const flush = () => new Promise(resolve => setImmediate(resolve));
const settings = { consent: true, language: "fr", allowBrowser: true };
const result = (text, isFinal = true) => Object.assign([{ transcript: text }], { isFinal });

function fixture(options = {}) {
  const events = [], answers = [], asks = [], mics = [], spoken = [], timers = new Map();
  let now = 0, next = 0, playing = null;
  const clock = {
    setTimeout(fn, delay) { timers.set(++next, [now + delay, fn]); return next; },
    clearTimeout(id) { timers.delete(id); },
    tick(ms) {
      const target = now + ms;
      while (timers.size) {
        const [id, [at, fn]] = [...timers.entries()].sort((a, b) => a[1][0] - b[1][0])[0];
        if (at > target) break;
        now = at;
        timers.delete(id);
        fn();
      }
      now = target;
    },
  };
  const finishVoice = (outcome = "played") => { playing?.(outcome); playing = null; };
  const session = new VoiceSession({
    clock, pause: 10, listenTimeout: 100, endTimeout: 30,
    connect: async signal => { events.push(["connect", signal]); },
    disconnect: () => events.push(["disconnect"]),
    text: key => key,
    recognition: current => {
      assert.equal(playing, null, "microphone must not open during speech");
      const mic = {
        settings: current, start() { this.started = true; }, stop() { this.stopping = true; },
        abort() { this.aborted = true; },
        finalize(...items) { this.onresult?.({ results: items, resultIndex: 0 }); },
        end() { this.onend?.(); }, fail(error) { this.onerror?.({ error }); },
      };
      mics.push(mic);
      return mic;
    },
    speaker: {
      play(text, current) {
        assert.equal(playing, null, "speech must not overlap");
        assert.equal(session.record, null, "speech must not start with an active microphone");
        spoken.push([text, current.language]);
        return new Promise(resolve => { playing = resolve; });
      },
      cancel() { finishVoice("cancelled"); },
    },
    ask(question, current, signal) {
      assert.equal(session.record, null);
      return new Promise((resolve, reject) => asks.push({ question, current, signal, resolve, reject }));
    },
    update: event => events.push([event.state, event.note]),
    announce: message => events.push(["announce", message]),
    answer: value => answers.push(value),
    ...options,
  });
  async function begin() {
    const start = session.start(settings);
    await flush();
    assert.deepEqual(spoken[0], ["readyPrompt", "fr"]);
    assert.equal(mics.length, 0);
    finishVoice();
    await start;
    clock.tick(10);
    assert.equal(session.state, State.LISTENING);
    return mics.at(-1);
  }
  return { session, events, clock, answers, asks, mics, spoken, finishVoice, begin };
}

test("startup requires consent and waits for connection", async () => {
  const f = fixture();
  await f.session.start({ ...settings, consent: false });
  assert.deepEqual(f.events, [[State.ERROR, "consentRequired"], ["announce", "consentRequired"]]);
  let connect;
  f.session.connect = () => new Promise(resolve => { connect = resolve; });
  const start = f.session.start(settings);
  assert.equal(f.session.state, State.CONNECTING);
  assert.deepEqual(f.spoken, []);
  connect();
  await flush();
  f.finishVoice();
  await start;
  f.clock.tick(10);
  assert.equal(f.mics[0].settings.language, "fr");
});

for (const reason of ["pairRequired", "cameraDenied", "cameraUnavailable", "connectionLost"]) {
  test(`startup failure releases resources: ${reason}`, async () => {
    const f = fixture({ connect: async () => { throw { reason }; } });
    await f.session.start(settings);
    assert.deepEqual(f.events.slice(-3), [["disconnect"], [State.ERROR, reason], ["announce", reason]]);
    assert.deepEqual(f.asks, []);
  });
}

test("final speech sends one question after the mic closes, speaks the answer, then resumes", async () => {
  const f = fixture();
  const mic = await f.begin();
  mic.finalize(result("interim", false));
  mic.finalize(result("  "));
  assert.equal(mic.stopping, undefined);
  mic.finalize(result("Que vois-tu ?"));
  assert.equal(f.session.state, State.PROCESSING);
  mic.finalize(result("duplicate"));
  assert.equal(f.asks.length, 0);
  mic.end();
  mic.end();
  assert.equal(f.asks.length, 1);
  assert.equal(f.asks[0].question, "Que vois-tu ?");
  assert.equal(f.session.askTyped("another"), false);
  f.asks[0].resolve({ answer: "Un livre." });
  await flush();
  assert.equal(f.session.state, State.SPEAKING);
  assert.deepEqual(f.spoken.at(-1), ["Un livre.", "fr"]);
  f.clock.tick(100);
  assert.equal(f.mics.length, 1);
  f.finishVoice();
  await flush();
  f.clock.tick(10);
  assert.equal(f.mics.length, 2);
  assert.equal(f.session.state, State.LISTENING);
});

test("silence retries once; Talk Again and typed questions remain available", async () => {
  const f = fixture();
  const first = await f.begin();
  f.clock.tick(100);
  assert.equal(first.stopping, true);
  first.end();
  f.clock.tick(10);
  f.mics[1].fail("no-speech");
  f.mics[1].end();
  assert.equal(f.spoken.at(-1)[0], "micRetry");
  f.finishVoice();
  await flush();
  f.clock.tick(100);
  assert.equal(f.mics.length, 2);
  f.session.talkAgain();
  assert.equal(f.mics.length, 3);
  assert.equal(f.session.askTyped(" A typed question "), true);
  f.mics[2].end();
  assert.equal(f.asks[0].question, "A typed question");
});

test("mic denial, stalled shutdown and playback failure never auto-restart", async () => {
  const f = fixture();
  const first = await f.begin();
  first.fail("not-allowed");
  first.end();
  assert.equal(f.spoken.at(-1)[0], "micDenied");
  f.finishVoice("unavailable");
  await flush();
  f.clock.tick(300);
  assert.equal(f.mics.length, 1);
  f.session.talkAgain();
  f.mics[1].finalize(result("stalled browser"));
  f.clock.tick(30);
  assert.equal(f.mics[1].aborted, true);
  assert.equal(f.asks.length, 0);
  assert.equal(f.session.state, State.ERROR);
});

for (const phase of ["listening", "processing", "speaking"]) {
  test(`Stop cancels ${phase} and prevents stale callbacks`, async () => {
    const f = fixture();
    const mic = await f.begin();
    const lateResult = mic.onresult;
    if (phase !== "listening") {
      mic.finalize(result("question"));
      mic.end();
    }
    if (phase === "speaking") {
      f.asks[0].resolve({ answer: "short answer" });
      await flush();
    }
    f.session.stop();
    lateResult({ results: [result("late result")] });
    if (phase === "processing") f.asks[0].resolve({ answer: "stale answer" });
    await flush();
    f.clock.tick(1000);
    assert.equal(f.session.state, State.STOPPED);
    assert.equal(f.events.filter(([event]) => event === "disconnect").length, 1);
    assert.equal(f.mics.length, 1);
    assert.equal(f.answers.includes("stale answer"), false);
    if (phase === "listening") assert.equal(mic.aborted, true);
    else assert.equal(f.asks[0].signal.aborted, true);
  });
}

test("service rejection is spoken without resending; expired pairing stops the session", async () => {
  const f = fixture();
  const mic = await f.begin();
  mic.finalize(result("camera question"));
  mic.end();
  f.asks[0].reject({ reason: "noLiveFrame" });
  await flush();
  assert.deepEqual(f.spoken.at(-1), ["noLiveFrame", "fr"]);
  f.finishVoice();
  await flush();
  f.clock.tick(10);
  assert.equal(f.asks.length, 1);
  f.mics[1].finalize(result("another question"));
  f.mics[1].end();
  f.asks[1].reject({ status: 401 });
  await flush();
  assert.deepEqual(f.events.at(-1), ["announce", "pairRequired"]);
  assert.equal(f.asks[1].signal.aborted, true);
});

test("Repeat Answer pauses the microphone without making another provider request", async () => {
  const f = fixture();
  const mic = await f.begin();
  mic.finalize(result("question"));
  mic.end();
  f.asks[0].resolve({ answer: "short answer" });
  await flush();
  f.finishVoice();
  await flush();
  f.clock.tick(10);
  f.session.repeat();
  f.session.repeat();
  f.mics.at(-1).end();
  assert.deepEqual(f.spoken.at(-1), ["short answer", "fr"]);
  assert.equal(f.asks.length, 1);
  f.session.stopSpeaking();
  await flush();
  f.clock.tick(10);
  assert.equal(f.session.state, State.LISTENING);
});
