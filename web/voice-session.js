export const State = Object.freeze({
  IDLE: "IDLE", CONNECTING: "CONNECTING", READY: "READY", LISTENING: "LISTENING",
  PROCESSING: "PROCESSING", SPEAKING: "SPEAKING", ERROR: "ERROR", STOPPED: "STOPPED",
});

const quietErrors = new Set(["", "no-speech", "aborted"]);
const permissionErrors = new Set(["not-allowed", "service-not-allowed"]);
const idleStates = new Set([State.READY, State.LISTENING, State.ERROR]);

export class VoiceSession {
  constructor({ connect, disconnect, recognition, speaker, ask, update, announce, answer, text,
    clock = globalThis, pause = 450, listenTimeout = 20000, endTimeout = 2500 }) {
    Object.assign(this, { connect, disconnect, recognition, speaker, ask, update, announce, answer, text,
      clock, pause, listenTimeout, endTimeout });
    this.run = this.record = this.timer = this.lastAnswer = null;
    this.state = State.IDLE;
  }

  set(state, note) {
    this.state = state;
    this.update({ state, note, active: Boolean(this.run), hasAnswer: Boolean(this.lastAnswer) });
  }

  async start(settings) {
    if (this.run) return;
    if (!settings.consent) return this.stop("consentRequired");
    const run = this.run = { settings, abort: new AbortController(), misses: 0 };
    this.set(State.CONNECTING, "connecting");
    try {
      await this.connect(run.abort.signal);
    } catch (error) {
      if (run === this.run) this.stop(error.reason || "connectionLost");
      return;
    }
    if (run === this.run) {
      this.set(State.READY, "readyState");
      await this.play(run, this.text("readyPrompt"), { readyPrompt: true });
    }
  }

  stop(reason = "stopped") {
    const run = this.run;
    this.run = null;
    run?.abort.abort();
    this.clearRestart();
    const record = this.record;
    if (record) this.release(record, true);
    if (run) {
      this.speaker.cancel();
      this.disconnect();
    }
    this.lastAnswer = null;
    this.set(reason === "stopped" ? State.STOPPED : State.ERROR, reason);
    this.announce(this.text(reason));
  }

  clearRestart() {
    this.clock.clearTimeout(this.timer);
    this.timer = null;
  }

  schedule(run) {
    this.clearRestart();
    this.set(State.READY, "readyState");
    this.timer = this.clock.setTimeout(() => {
      this.timer = null;
      if (this.run === run && this.state === State.READY) this.listen(run);
    }, this.pause);
  }

  talkAgain() {
    if (!this.run || ![State.READY, State.ERROR].includes(this.state)) return;
    this.run.misses = 0;
    this.listen(this.run);
  }

  listen(run) {
    if (this.run !== run || this.record || ![State.READY, State.ERROR].includes(this.state)) return;
    this.clearRestart();
    let native;
    try { native = this.recognition(run.settings); } catch { return this.micError(run, "micRetry"); }
    if (!native) {
      this.set(State.ERROR, "micUnavailable");
      this.announce(this.text("micUnavailable"));
      return;
    }
    const record = this.record = { run, native, error: "", next: null, ending: false };
    this.set(State.LISTENING, "listening");
    native.onresult = event => {
      if (this.record !== record || record.ending) return;
      const phrase = Array.from(event.results).slice(event.resultIndex ?? 0)
        .filter(result => result.isFinal).map(result => result[0]?.transcript || "").join(" ").trim().slice(0, 800);
      if (phrase) this.end(record, { question: phrase });
    };
    native.onerror = event => {
      if (this.record !== record || record.next) return;
      record.error = event.error || "unknown";
      this.end(record);
    };
    native.onend = () => {
      if (this.record !== record) return;
      this.release(record);
      if (record.next?.question) void this.process(run, record.next.question);
      else if (record.next?.repeat) void this.play(run, this.lastAnswer);
      else if (permissionErrors.has(record.error)) this.micError(run, "micDenied");
      else if (!quietErrors.has(record.error) || ++run.misses >= 2) this.micError(run, "micRetry");
      else this.schedule(run);
    };
    record.timeout = this.clock.setTimeout(() => this.end(record), this.listenTimeout);
    try { native.start(); } catch {
      if (this.record === record) {
        this.release(record, true);
        this.micError(run, "micRetry");
      }
    }
  }

  end(record, next = null) {
    if (this.record !== record || record.ending) return;
    record.ending = true;
    record.next = next;
    if (next) this.set(State.PROCESSING, next.question ? "processing" : "readyState");
    this.clock.clearTimeout(record.timeout);
    record.endTimer = this.clock.setTimeout(() => {
      if (this.record !== record) return;
      this.release(record, true);
      // Do not speak or submit if the browser never confirmed the microphone closed.
      const reason = permissionErrors.has(record.error) ? "micDenied" : "micRetry";
      this.set(State.ERROR, reason);
      this.announce(this.text(reason));
    }, this.endTimeout);
    try { record.native.stop(); } catch { /* Wait for end or the watchdog. */ }
  }

  release(record, abort = false) {
    this.record = null;
    this.clock.clearTimeout(record.timeout);
    this.clock.clearTimeout(record.endTimer);
    record.native.onend = record.native.onresult = record.native.onerror = null;
    if (abort) {
      try { record.native.abort(); } catch {}
    }
  }

  micError(run, reason) {
    if (run === this.run) void this.play(run, this.text(reason), { error: reason });
  }

  askTyped(value) {
    const question = value.trim().slice(0, 800);
    if (!question || !this.run || !idleStates.has(this.state)) return false;
    if (this.record?.ending) return false;
    this.clearRestart();
    if (this.record) this.end(this.record, { question });
    else void this.process(this.run, question);
    return true;
  }

  repeat() {
    if (!this.lastAnswer || !this.run || !idleStates.has(this.state) || this.record?.ending) return;
    this.clearRestart();
    if (this.record) this.end(this.record, { repeat: true });
    else void this.play(this.run, this.lastAnswer);
  }

  stopSpeaking() {
    if (this.state === State.SPEAKING) this.speaker.cancel();
  }

  async process(run, question) {
    if (run !== this.run) return;
    run.misses = 0;
    this.set(State.PROCESSING, "processing");
    let result;
    try {
      result = await this.ask(question, run.settings, run.abort.signal);
    } catch (error) {
      if (run !== this.run) return;
      if (error.status === 401 || error.status === 403) return this.stop("pairRequired");
      const reason = error.reason || "answerUnavailable";
      await this.play(run, this.text(reason), { note: reason });
      return;
    }
    if (run !== this.run) return;
    if (typeof result?.answer !== "string" || !result.answer.trim()) {
      await this.play(run, this.text("answerUnavailable"));
      return;
    }
    this.lastAnswer = result.answer;
    this.answer(result.answer);
    await this.play(run, this.lastAnswer);
  }

  async play(run, phrase, { error, note, readyPrompt } = {}) {
    if (run !== this.run) return;
    this.set(State.SPEAKING, readyPrompt ? "readyPrompt" : (note || error || "speaking"));
    let result = "unavailable";
    try { result = await this.speaker.play(phrase, run.settings); } catch {}
    if (run !== this.run) return;
    if (error || result === "unavailable") {
      const reason = error || "speechUnavailable";
      this.set(State.ERROR, reason);
      if (result !== "played") {
        this.announce(result === "unavailable" && !error ? `${phrase} ${this.text(reason)}` : this.text(reason));
      }
    } else this.schedule(run);
  }
}
