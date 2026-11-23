const CACHE = "visual-camera-shell-v3";
const SHELL = [
  "/phone", "/ui/style.css", "/ui/phone.js", "/ui/common.js", "/ui/icons.svg",
  "/ui/assistant.js", "/ui/voice.js",
  "/ui/voice-session.js", "/ui/phone-camera.js", "/ui/mobile-strings.js", "/ui/mobile-question.js",
  "/ui/manifest.webmanifest", "/ui/icon-192.png", "/ui/icon-512.png",
];
self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL)));
});
self.addEventListener("activate", event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(
    keys.filter(key => key.startsWith("visual-camera-shell-") && key !== CACHE)
      .map(key => caches.delete(key))
  )).then(() => self.clients.claim()));
});
self.addEventListener("fetch", event => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== self.location.origin || !SHELL.includes(url.pathname)) return;
  event.respondWith(fetch(event.request).catch(async () => {
    const cache = await caches.open(CACHE);
    return await cache.match(url.pathname) || Response.error();
  }));
});
