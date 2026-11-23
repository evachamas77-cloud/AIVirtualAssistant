const stopped = () => new DOMException("Camera stopped", "AbortError");
const unavailable = reason => Object.assign(new Error(reason), { reason });

function awaitConnection(pc, signal) {
  return new Promise((resolve, reject) => {
    const finish = error => {
      pc.removeEventListener("connectionstatechange", changed);
      signal.removeEventListener("abort", aborted);
      if (error) reject(error);
      else resolve();
    };
    const aborted = () => finish(stopped());
    const changed = () => {
      if (pc.connectionState === "connected") finish();
      else if (["failed", "closed"].includes(pc.connectionState)) finish(unavailable("connectionLost"));
    };
    pc.addEventListener("connectionstatechange", changed);
    signal.addEventListener("abort", aborted, { once: true });
    if (signal.aborted) aborted();
    else changed();
  });
}

export class PhoneCamera {
  constructor({ getMedia, createPeer, negotiate, closePeer, changed, lost, label }) {
    Object.assign(this, { getMedia, createPeer, negotiate, closePeer, changed, lost, label });
    this.stream = this.pending = this.peer = this.lossTimer = null;
    this.stage = "cameraOff";
  }

  state(stage) {
    this.stage = stage;
    this.changed();
  }

  async open(signal) {
    signal?.throwIfAborted();
    if (this.stream) return;
    if (!this.getMedia || this.pending) throw unavailable("cameraUnavailable");
    const attempt = this.pending = {};
    this.state("cameraPermission");
    try {
      const stream = await this.getMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280, max: 1280 }, height: { ideal: 720, max: 1280 },
          frameRate: { ideal: 15, max: 20 },
        },
      });
      if (signal?.aborted || this.pending !== attempt || !stream.getVideoTracks()[0]) {
        stream.getTracks().forEach(track => track.stop());
        if (signal?.aborted || this.pending !== attempt) throw stopped();
        throw unavailable("cameraUnavailable");
      }
      this.stream = stream;
      stream.getVideoTracks()[0].addEventListener("ended", () => {
        if (this.stream === stream) this.lost("cameraUnavailable");
      });
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw unavailable(error.name === "NotAllowedError" ? "cameraDenied" : "cameraUnavailable");
    } finally {
      if (this.pending === attempt) {
        this.pending = null;
        this.state(this.stream ? "cameraReady" : "cameraOff");
      }
    }
  }

  async streamToServer(signal) {
    signal?.throwIfAborted();
    if (!this.stream || !this.createPeer) throw unavailable("connectionLost");
    if (this.peer?.pc.connectionState === "connected") return;
    if (this.peer) throw unavailable("connectionLost");
    const peer = this.peer = this.createPeer();
    const timer = setTimeout(() => peer.abort.abort(), 18000);
    const cancel = () => peer.abort.abort();
    signal?.addEventListener("abort", cancel, { once: true });
    this.state("connecting");
    try {
      peer.pc.addTransceiver(this.stream.getVideoTracks()[0], { direction: "sendonly", streams: [this.stream] });
      await this.negotiate(peer, "camera", this.label);
      await awaitConnection(peer.pc, peer.abort.signal);
      if (this.peer !== peer) throw stopped();
      this.watch(peer);
      this.state("streaming");
    } catch (error) {
      if (signal?.aborted || this.peer !== peer) throw stopped();
      this.close();
      if (error.status === 401 || error.status === 403) throw unavailable("pairRequired");
      if (error.status === 409) throw unavailable("cameraInUse");
      throw unavailable("connectionLost");
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", cancel);
    }
  }

  watch(peer) {
    peer.pc.addEventListener("connectionstatechange", () => {
      if (this.peer !== peer) return;
      const state = peer.pc.connectionState;
      clearTimeout(this.lossTimer);
      if (state === "connected") this.state("streaming");
      if (state === "disconnected") {
        this.state("reconnecting");
        this.lossTimer = setTimeout(() => {
          if (this.peer === peer && peer.pc.connectionState === "disconnected") this.lost("connectionLost");
        }, 6000);
      }
      if (state === "failed" || state === "closed") this.lost("connectionLost");
    });
  }

  close() {
    clearTimeout(this.lossTimer);
    const peer = this.peer;
    this.peer = this.pending = null;
    if (peer) this.closePeer(peer);
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.state("cameraOff");
  }
}
