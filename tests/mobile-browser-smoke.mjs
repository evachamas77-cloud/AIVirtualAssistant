// Optional desktop Chrome smoke test: node tests/mobile-browser-smoke.mjs
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

const chrome = process.env.CHROME_EXE || join(process.env.ProgramFiles || "C:/Program Files", "Google/Chrome/Application/chrome.exe");
const python = process.env.PYTHON_EXE || resolve(".venv/Scripts/python.exe");
const root = await mkdtemp(join(tmpdir(), "visual-assistant-smoke-"));
const code = "mobile-smoke-pairing-code";
const processes = [];
let socket;

function launch(command, args, options, pattern) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { windowsHide: true, ...options });
    processes.push(child);
    const deadline = setTimeout(() => reject(new Error(`Timed out launching ${command}`)), 15000);
    let startup = "";
    const line = chunk => {
      startup = (startup + chunk).slice(-12000);
      const match = startup.match(pattern);
      if (match) { clearTimeout(deadline); resolve(match[1]); }
    };
    child.stdout.on("data", line);
    child.stderr.on("data", line);
    child.on("error", error => { clearTimeout(deadline); reject(error); });
    child.on("exit", status => { clearTimeout(deadline); reject(new Error(`Startup exited: ${status}\n${startup}`)); });
  });
}

try {
  const server = await launch(python, ["-u", "-c", `
from backend import main
import uvicorn
main.detect_objects = lambda frame: []
main.recognize_text = lambda frame: ("test text", [])
uvicorn.run(main.app, host="127.0.0.1", port=0, access_log=False)
`], { env: { ...process.env, OPENAI_API_KEY: "", FRAME_STORAGE_DIR: join(root, "archive"),
    LIVE_SESSION_CODE: code, FRAME_MIN_SHARPNESS: "0", SEMANTIC_ENABLED: "0" } },
  /Uvicorn running on (http:\/\/127\.0\.0\.1:\d+)/);
  const debug = await launch(chrome, ["--headless=new", "--no-first-run", "--remote-debugging-port=0",
    `--user-data-dir=${join(root, "chrome")}`, "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    "--disable-features=WebRtcHideLocalIpsWithMdns", "about:blank"], {}, /DevTools listening on ws:\/\/(\S+)\/devtools/);
  const [page] = await fetch(`http://${debug}/json/list`).then(response => response.json());
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });

  let id = 0;
  const pending = new Map(), errors = [];
  socket.addEventListener("message", event => {
    const message = JSON.parse(event.data);
    if (message.method === "Runtime.exceptionThrown") errors.push(message.params.exceptionDetails.exception?.description);
    pending.get(message.id)?.(message);
  });
  function call(method, params = {}) {
    return new Promise((resolve, reject) => {
      const seq = ++id;
      const timeout = setTimeout(() => { pending.delete(seq); reject(new Error(`CDP timeout: ${method}`)); }, 15000);
      pending.set(seq, message => {
        pending.delete(seq);
        clearTimeout(timeout);
        if (message.error) reject(new Error(message.error.message));
        else resolve(message.result);
      });
      socket.send(JSON.stringify({ id: seq, method, params }));
    });
  }
  async function evaluate(expression) {
    const result = await call("Runtime.evaluate", { expression, returnByValue: true, userGesture: true });
    if (result.exceptionDetails) throw new Error(result.result.description);
    return result.result.value;
  }
  async function until(expression) {
    for (let i = 0; i < 120; i++) {
      if (await evaluate(expression)) return;
      await delay(100);
    }
    throw new Error(`Not reached: ${expression}; URL: ${await evaluate("location.href")}; ` +
      `screen: ${await evaluate("document.body?.innerText")}; errors: ${errors.join("; ")}`);
  }

  await call("Page.enable");
  await call("Runtime.enable");
  await call("Emulation.setDeviceMetricsOverride", { width: 360, height: 760, deviceScaleFactor: 1, mobile: true });
  await call("Page.addScriptToEvaluateOnNewDocument", { source: `
    window.spoken = []; window.requests = []; window.recognizer = null;
    window.SpeechRecognition = class {
      static async available() { return 'available'; }
      start() { if (window.recognizer) throw new Error('Overlapping mic'); window.recognizer = this; }
      stop() { window.recognizer = null; queueMicrotask(() => this.onend?.()); }
      abort() { window.recognizer = null; }
    };
    window.SpeechSynthesisUtterance = class { constructor(text) { this.text = text; } };
    Object.defineProperty(window, 'speechSynthesis', { value: {
      getVoices: () => [{ lang: 'en-US', localService: true }, { lang: 'fr-FR', localService: true }],
      speak(utterance) {
        if (window.recognizer) throw new Error('Spoke over mic');
        spoken.push([utterance.text, utterance.lang]);
        setTimeout(() => { utterance.onstart?.(); utterance.onend?.(); }, 50);
      }, cancel() {},
    }});
    const original = window.fetch;
    window.fetch = (url, options) => {
      if (url !== '/ai/ask') return original(url, options);
      requests.push(JSON.parse(options.body));
      return Promise.resolve(new Response(JSON.stringify({ answer: 'Simulated answer ' + requests.length }), {
        headers: { 'Content-Type': 'application/json' },
      }));
    };
    window.say = transcript => recognizer.onresult({ results: [Object.assign([{ transcript }], { isFinal: true })] });
  ` });
  const navigation = await call("Page.navigate", { url: server + "/phone" });
  assert.equal(navigation.errorText, undefined);
  await until("document.querySelector('#pair-form')?.hidden === false && document.querySelector('#ai-ask-form') !== null");
  assert.equal(await evaluate("document.documentElement.scrollWidth === 360"), true);
  await evaluate(`document.querySelector('#pair-code').value = ${JSON.stringify(code)}; document.querySelector('#pair-form').requestSubmit()`);
  await until("document.querySelector('#pair-form').hidden");
  await evaluate(`document.querySelector('#session-consent').click(); document.querySelector('#start-assistant').click()`);
  await until("window.recognizer !== null && document.querySelector('#voice-state').dataset.state === 'LISTENING'");
  await evaluate("window.say('What is in front of me?')");
  await until("window.recognizer !== null && window.spoken.length === 2");
  await evaluate("window.say('When did I last see a laptop?')");
  await until("window.recognizer !== null && window.spoken.length === 3");
  const bodies = await evaluate("requests");
  assert.deepEqual(bodies.map(item => [item.scope, item.live_only, item.consent]), [
    ["current", true, true], ["history", undefined, true],
  ]);
  await until("document.querySelector('#overall').textContent.includes('Camera streaming')");
  const response = await fetch(server + "/frames?limit=1").then(reply => reply.json());
  assert.equal(response.items.length, 1);
  await evaluate("document.querySelector('#stop-assistant').click()");
  assert.equal(await evaluate("!document.querySelector('#session-consent').checked && recognizer === null"), true);
  await until("navigator.serviceWorker.controller !== null");
  const manifest = await call("Page.getAppManifest");
  assert.equal(manifest.errors.filter(error => error.critical).length, 0);
  assert.equal(JSON.parse(manifest.data).start_url, "/phone");
  await call("Emulation.clearDeviceMetricsOverride");
  await call("Page.navigate", { url: server + "/" });
  await until("document.querySelector('#ai-ask-form') !== null && document.title.includes('Live monitor')");
  assert.equal(await evaluate("document.querySelector('#start-assistant') === null"), true);
  assert.deepEqual(errors, []);
  console.log("Desktop Chrome smoke passed: mobile layout, pairing, real browser WebRTC, archive, mocked voice turns, manifest, and dashboard.");
  await call("Browser.close").catch(() => {});
} finally {
  socket?.close();
  const exits = processes.map(child => new Promise(resolve => {
    if (child.exitCode !== null || child.signalCode !== null || !child.pid) return resolve();
    child.once("exit", resolve);
    child.kill();
  }));
  await Promise.all(exits);
  if (resolve(root).startsWith(resolve(tmpdir(), "visual-assistant-smoke-"))) {
    await rm(root, { recursive: true, force: true, maxRetries: 6, retryDelay: 150 });
  }
}
