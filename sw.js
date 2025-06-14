const CACHE_NAME = 'indic-lang-detect-v1';
const urlsToCache = [
  '/',
  '/static/icon.png',
  '/templates/index.html',
  '/static/sw.js',
  '/manifest.json'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});