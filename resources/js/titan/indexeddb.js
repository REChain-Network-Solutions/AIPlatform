/* Titan Zero BOS IndexedDB bootstrap */

const DB_NAME = 'titan-zero-bos';
const DB_VERSION = 1;
const STORES = ['local_signals', 'sync_queue', 'runtime_meta'];

export function openTitanDb() {
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

export async function saveLocalSignal(signal) {
  const db = await openTitanDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('local_signals', 'readwrite');
    tx.objectStore('local_signals').add({
      ...signal,
      status: 'pending',
      queued_at: Date.now(),
    });
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
  });
}

export async function readPendingSignals(limit = 50) {
  const db = await openTitanDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('local_signals', 'readonly');
    const store = tx.objectStore('local_signals');
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

export async function markSignalStatus(id, status = 'dispatched') {
  const db = await openTitanDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('local_signals', 'readwrite');
    const store = tx.objectStore('local_signals');
    const getReq = store.get(id);
    getReq.onsuccess = () => {
      const record = getReq.result;
      if (!record) return resolve(false);
      record.status = status;
      record.updated_at = Date.now();
      store.put(record);
    };
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
  });
}

export async function logRuntimeMeta(metaKey, metaValue) {
  const db = await openTitanDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('runtime_meta', 'readwrite');
    tx.objectStore('runtime_meta').add({
      meta_key: metaKey,
      meta_value: metaValue,
      created_at: Date.now(),
    });
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
  });
}
