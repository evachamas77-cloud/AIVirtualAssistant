import assert from "node:assert/strict";
import test from "node:test";
import { answerError, questionSource } from "../web/mobile-question.js";
import { strings } from "../web/mobile-strings.js";
import { PhoneCamera } from "../web/phone-camera.js";

const flush = () => new Promise(resolve => setImmediate(resolve));

function cameraFixture(options = {}) {
  const track = Object.assign(new EventTarget(), { stop() { this.stopped = true; } });
  const stream = { getVideoTracks: () => [track], getTracks: () => [track] };
  const pc = Object.assign(new EventTarget(), {
    connectionState: "new",
    addTransceiver(video, config) { this.transceiver = [video, config]; },
    setState(state) { this.connectionState = state; this.dispatchEvent(new Event("connectionstatechange")); },
  });
  const peer = { pc, abort: new AbortController() };
  const offers = [];
  const lost = [];
  const camera = new PhoneCamera({
    async getMedia(constraints) { assert.equal(constraints.audio, false); return stream; },
    createPeer: () => peer,
    negotiate: async (...args) => { offers.push(args); },
    closePeer: session => { session.abort.abort(); session.pc.setState("closed"); },
    changed: () => {}, lost: reason => { lost.push(reason); camera.close(); }, label: "Android phone",
    ...options,
  });
  return { camera, peer, stream, track, pc, offers, lost };
}

test("video-only camera waits for WebRTC connection and uses existing signaling", async () => {
  const { camera, stream, track, pc, offers, peer, lost } = cameraFixture();
  await camera.open();
  const connection = camera.streamToServer();
  await flush();
  assert.equal(camera.stage, "connecting");
  assert.deepEqual(pc.transceiver, [track, { direction: "sendonly", streams: [stream] }]);
  assert.deepEqual(offers[0], [peer, "camera", "Android phone"]);
  pc.setState("connected");
  await connection;
  assert.equal(camera.stage, "streaming");
  pc.setState("disconnected");
  assert.equal(camera.stage, "reconnecting");
  pc.setState("failed");
  assert.deepEqual(lost, ["connectionLost"]);
  assert.equal(track.stopped, true);
  assert.equal(peer.abort.signal.aborted, true);
  assert.equal(camera.stage, "cameraOff");
});

test("stop cancels a pending camera permission result without retaining a camera track", async () => {
  let finish;
  const { camera, stream, track } = cameraFixture({ getMedia: () => new Promise(resolve => { finish = resolve; }) });
  const abort = new AbortController();
  const pending = camera.open(abort.signal);
  assert.equal(camera.stage, "cameraPermission");
  abort.abort();
  camera.close();
  finish(stream);
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(track.stopped, true);
  assert.equal(camera.stream, null);
});

test("camera denial and expired pairing are recoverable startup errors", async () => {
  const denied = cameraFixture({ getMedia: async () => { throw new DOMException("denied", "NotAllowedError"); } });
  await assert.rejects(denied.camera.open(), { reason: "cameraDenied" });
  assert.equal(denied.camera.stage, "cameraOff");
  const expired = cameraFixture({ negotiate: async () => { throw { status: 401 }; } });
  await expired.camera.open();
  await assert.rejects(expired.camera.streamToServer(), { reason: "pairRequired" });
  assert.equal(expired.track.stopped, true);
  assert.equal(expired.peer.abort.signal.aborted, true);
});

test("current scenes use live-only; common English and French questions use archived evidence", () => {
  for (const question of ["What is in front of me?", "Read the text on this paper", "Que vois-tu ?"]) {
    assert.deepEqual(questionSource(question), { scope: "current", live_only: true });
  }
  for (const question of ["When did I last see a laptop?", "Have I seen a notebook before?",
    "Find the last time I saw a red car", "Quand ai-je vu un canap\u00e9 ?", "Ai-je d\u00e9j\u00e0 vu un ordinateur ?"]) {
    assert.deepEqual(questionSource(question), { scope: "history" });
  }
  assert.equal(answerError(404, "history"), "noHistory");
  assert.equal(answerError(404, "current"), "noLiveFrame");
  assert.equal(answerError(undefined, "current"), "connectionLost");
});

test("EN/FR relative dates filter by the device's local day", () => {
  const now = new Date(2026, 10, 22, 14);
  const yesterday = new Date(2026, 10, 21);
  const end = new Date(2026, 10, 22).getTime() - 1;
  for (const question of ["What did I see yesterday?", "Qu'est-ce que j'ai vu hier ?"]) {
    assert.deepEqual(questionSource(question, now), {
      scope: "history", start: yesterday.toISOString(), end: new Date(end).toISOString().slice(0, -1) + "999Z",
    });
  }
  assert.deepEqual(Object.keys(strings.fr).sort(), Object.keys(strings.en).sort());
  assert.notEqual(strings.fr.readyPrompt, strings.en.readyPrompt);
  assert.notEqual(strings.fr.micDenied, strings.en.micDenied);
});
