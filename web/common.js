export const byId = id => document.getElementById(id);

export function setStatus(label, state = "idle") {
  const status = byId("overall");
  status.dataset.state = state;
  status.lastElementChild.textContent = label;
}

export function notice(message = "") {
  const element = byId("notice");
  element.textContent = message;
  element.hidden = !message;
}

export async function api(path, options) {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(typeof body.detail === "string" ? body.detail : "Request failed.");
    error.status = response.status;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

export function setupPairing(onPair, { show = notice, describe = error => error.message,
  insecure = () => "Trusted HTTPS is required to pair this device." } = {}) {
  const form = byId("pair-form");
  const input = byId("pair-code");
  const button = form.querySelector("button");
  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (!window.isSecureContext) {
      show(insecure());
      return;
    }
    button.disabled = true;
    show();
    try {
      await api("/live/session", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: input.value }),
      });
      input.value = "";
      await onPair();
    } catch (error) {
      show(describe(error));
    } finally {
      button.disabled = false;
    }
  });
}

export function newPeer() {
  return {
    pc: new RTCPeerConnection({ iceServers: [] }),
    abort: new AbortController(),
    id: null,
  };
}

function gather(pc, signal) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const settle = error => {
      clearTimeout(timer);
      pc.removeEventListener("icegatheringstatechange", changed);
      signal.removeEventListener("abort", aborted);
      if (error) reject(error);
      else resolve();
    };
    const changed = () => {
      if (pc.iceGatheringState === "complete") settle();
    };
    const aborted = () => settle(new DOMException("Connection stopped", "AbortError"));
    const timer = setTimeout(() => settle(new Error("ICE gathering timed out.")), 8000);
    pc.addEventListener("icegatheringstatechange", changed);
    signal.addEventListener("abort", aborted, { once: true });
    if (signal.aborted) aborted();
    else changed();
  });
}

export async function negotiate(peer, role, label) {
  const { pc, abort: { signal } } = peer;
  await pc.setLocalDescription(await pc.createOffer());
  await gather(pc, signal);
  const { sdp, type } = pc.localDescription;
  const answer = await api("/live/offer", {
    method: "POST", signal, headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp, type, role, label }),
  });
  peer.id = answer.peer_id;
  if (signal.aborted) {
    closePeer(peer);
    throw new DOMException("Connection stopped", "AbortError");
  }
  await pc.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
}

export function closePeer(peer) {
  if (!peer) return;
  peer.abort.abort();
  peer.pc.close();
  if (peer.id) {
    fetch(`/live/peers/${encodeURIComponent(peer.id)}`, {
      method: "DELETE", keepalive: true,
    }).catch(() => {});
    peer.id = null;
  }
}
