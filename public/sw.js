const CACHE_NAME = 'titan-zero-bos-cache-v2';
const STATIC_SHELL = ['/', '/offline', '/manifest.webmanifest'];
const OFFLINE_RESPONSE = new Response('You are offline. Please try again when connected.', {
  headers: { 'Content-Type': 'text/plain' },
});

// Dedicated SW queue database (separate from app IndexedDB) for transient replay payloads.
const QUEUE_DB = 'titan-zero-bos-sync';
const QUEUE_STORE = 'tz_sync_queue';

const openQueueDb = () =>
  new Promise((resolve, reject) => {
    const request = indexedDB.open(QUEUE_DB, 1);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(QUEUE_STORE)) {
        db.createObjectStore(QUEUE_STORE, { keyPath: 'id', autoIncrement: true });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

async function queueRequest(request) {
  const db = await openQueueDb();
  const serialized = await serializeBody(request.clone());
  const record = {
    url: request.url,
    method: request.method,
    headers: Array.from(request.headers.entries()),
    body: serialized.body,
    bodyType: serialized.type,
    contentType: serialized.contentType,
    queued_at: Date.now(),
  };

  const tx = db.transaction(QUEUE_STORE, 'readwrite');
  const store = tx.objectStore(QUEUE_STORE);
  store.add(record);

  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
  });
}

async function replayQueuedRequests() {
  const db = await openQueueDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(QUEUE_STORE, 'readwrite');
    const store = tx.objectStore(QUEUE_STORE);
    const queued = [];
    tx.onerror = () => reject(tx.error);

    const cursorRequest = store.openCursor();
    cursorRequest.onerror = () => reject(cursorRequest.error);
    cursorRequest.onsuccess = async (event) => {
      const cursor = event.target.result;
      if (!cursor) {
        resolve(queued.length);
        return;
      }

      const requestData = cursor.value;
      try {
        await fetch(requestData.url, {
          method: requestData.method,
          headers: requestData.headers,
          body: rebuildRequestBody(requestData),
        });
        store.delete(cursor.primaryKey);
        queued.push(requestData);
      } catch (error) {
        // Keep the record for the next sync attempt.
      }

      cursor.continue();
    };
  });
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_SHELL)),
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((key) => (key !== CACHE_NAME ? caches.delete(key) : null)))),
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const { request } = event;

  if (request.method === 'GET') {
    event.respondWith(
      caches.match(request).then((cached) => cached || fetch(request).then((response) => {
        const copy = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
        return response;
      }).catch(() => caches.match('/offline') || OFFLINE_RESPONSE)),
    );
    return;
  }

  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method)) {
    event.respondWith(
      fetch(request.clone()).catch(async () => {
        await queueRequest(request);
        if (self.registration?.sync) {
          await self.registration.sync.register('titan-sync-queue');
        }
        return OFFLINE_RESPONSE;
      }),
    );
  }
});

self.addEventListener('sync', (event) => {
  if (event.tag === 'titan-sync-queue') {
    event.waitUntil(replayQueuedRequests());
  }
});

async function serializeBody(request) {
  if (request.method === 'GET') {
    return { body: null, type: 'none', contentType: null };
  }

  const contentType = request.headers.get('Content-Type') || '';
  if (contentType.includes('application/json') || contentType.startsWith('text/')) {
    return { body: await request.text(), type: 'text', contentType };
  }

  const buffer = await request.arrayBuffer();
  return { body: buffer, type: 'arrayBuffer', contentType };
}

function rebuildRequestBody(requestData) {
  if (requestData.bodyType === 'arrayBuffer' && requestData.body) {
    return new Blob([requestData.body], { type: requestData.contentType || '' });
  }

  return requestData.body;
}
