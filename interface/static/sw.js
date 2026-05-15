// Yuki PWA service worker.
// Strategy:
//  - App shell (HTML, icons, manifest) → stale-while-revalidate so it opens offline.
//  - Third-party CDN (fonts, React, Babel) → cache-first with revalidate.
//  - API calls (/chat, /api/*) → network-only. Failures bubble up to the app so
//    the UI can show "can't reach Yuki" instead of a stale response.

const VERSION = 'yuki-v1';
const SHELL_CACHE = `${VERSION}-shell`;
const CDN_CACHE = `${VERSION}-cdn`;

const SHELL_ASSETS = [
  '/m',
  '/static/manifest.webmanifest',
  '/static/icon-180.png',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/icon-512-maskable.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((k) => !k.startsWith(VERSION)).map((k) => caches.delete(k))
    )).then(() => self.clients.claim())
  );
});

function isApiRequest(url) {
  return url.pathname === '/chat' ||
         url.pathname.startsWith('/api/') ||
         url.pathname.startsWith('/voice') ||
         url.pathname.startsWith('/tts');
}

function isShell(url) {
  return url.pathname === '/m' ||
         url.pathname.startsWith('/static/');
}

function isCDN(url) {
  return url.origin === 'https://unpkg.com' ||
         url.origin === 'https://fonts.googleapis.com' ||
         url.origin === 'https://fonts.gstatic.com';
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  // API calls: network-only. Don't mask failures — the UI needs to know.
  if (isApiRequest(url)) return;

  // App shell: stale-while-revalidate.
  if (isShell(url) && url.origin === self.location.origin) {
    event.respondWith((async () => {
      const cache = await caches.open(SHELL_CACHE);
      const cached = await cache.match(req);
      const fetchPromise = fetch(req).then((resp) => {
        if (resp && resp.ok) cache.put(req, resp.clone());
        return resp;
      }).catch(() => null);
      return cached || (await fetchPromise) || new Response('offline', { status: 503 });
    })());
    return;
  }

  // CDN: cache-first with revalidate in the background.
  if (isCDN(url)) {
    event.respondWith((async () => {
      const cache = await caches.open(CDN_CACHE);
      const cached = await cache.match(req);
      if (cached) {
        fetch(req).then((resp) => {
          if (resp && resp.ok) cache.put(req, resp.clone());
        }).catch(() => {});
        return cached;
      }
      try {
        const resp = await fetch(req);
        if (resp && resp.ok) cache.put(req, resp.clone());
        return resp;
      } catch {
        return new Response('offline', { status: 503 });
      }
    })());
    return;
  }

  // Everything else: default (pass through).
});
