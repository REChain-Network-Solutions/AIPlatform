/* Titan Zero BOS IndexedDB bootstrap */

const DB_NAME = 'titan-zero-bos';
const DB_VERSION = 2;
const STORES = [
  'tz_jobs',
  'tz_customers',
  'tz_invoices',
  'tz_local_signals',
  'tz_sync_queue',
  'tz_runtime_meta',
];

export function initTitanDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      STORES.forEach((store) => {
        if (!db.objectStoreNames.contains(store)) {
          db.createObjectStore(store, { keyPath: 'id', autoIncrement: true });
        }
      });
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
export const initDB = initTitanDB;

const runWrite = async (storeName, mutator) => {
  const db = await initTitanDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    const store = tx.objectStore(storeName);
    const request = mutator(store);
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
    request.onerror = () => reject(request.error);
  });
};

export async function putRecord(storeName, record) {
  return runWrite(storeName, (store) => store.put({ ...record, updated_at: Date.now() }));
}

export async function getRecord(storeName, key) {
  const db = await initTitanDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    const store = tx.objectStore(storeName);
    const req = store.get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    tx.onerror = () => reject(tx.error);
  });
}

export async function queueSignal(signal) {
  return runWrite('tz_local_signals', (store) =>
    store.add({
      ...signal,
      status: 'pending',
      queued_at: Date.now(),
    }),
  );
}

export async function flushSignals(limit = 50) {
  const pending = await _readPendingSignals(limit);

  await Promise.all(
    pending.map((signal) =>
      runWrite('tz_sync_queue', (store) =>
        store.add({
          // Fallback to 'unknown_signal' when type metadata is absent.
          object_type: signal.signal_type ?? 'unknown_signal',
          object_id: signal.id ?? null,
          action: signal.action ?? 'capture',
          status: 'pending',
          retry_count: 0,
          payload_json: signal.payload_json ?? signal.payload ?? null,
          created_at: Date.now(),
        }),
      ),
    ),
  );

  await Promise.all(
    pending.map((signal) =>
      runWrite('tz_local_signals', (store) =>
        store.put({ ...signal, status: 'queued', updated_at: Date.now() }),
      ),
    ),
  );

  return pending.length;
}

// Internal helper to keep pending-signal traversal private to this module.
async function _readPendingSignals(limit = 50) {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('tz_local_signals', 'readonly');
    const store = tx.objectStore('tz_local_signals');
    const results = [];
    const cursorReq = store.openCursor();
    cursorReq.onsuccess = (event) => {
      const cursor = event.target.result;
      if (cursor && results.length < limit && cursor.value.status === 'pending') {
        results.push(cursor.value);
        cursor.continue();
      } else {
        resolve(results);
      }
    };
    cursorReq.onerror = () => reject(cursorReq.error);
  });
}
