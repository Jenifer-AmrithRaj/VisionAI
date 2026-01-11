/* File: static/service_worker.js
   Purpose: Enables offline caching for VisionAI PWA
*/

const CACHE_NAME = "visionai-cache-v1";
const urlsToCache = [
  "/",
  "/manifest.json",
  "/static/css/style.css",
  "/static/icons/icon-192x192.png",
  "/static/icons/icon-512x512.png",
  "/static/visionai_logo.png"
];

// Install
self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(urlsToCache);
    })
  );
  console.log("✅ VisionAI service worker installed.");
});

// Activate
self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(name => {
          if (name !== CACHE_NAME) {
            console.log("🧹 Deleting old cache:", name);
            return caches.delete(name);
          }
        })
      );
    })
  );
  console.log("✅ VisionAI service worker activated.");
});

// Fetch
self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request).catch(() => caches.match("/"));
    })
  );
});
