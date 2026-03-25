import { flushSignals, initDB } from './indexeddb';

export async function setupBackgroundSync() {
  await initDB();

  if ('serviceWorker' in navigator && 'SyncManager' in window) {
    navigator.serviceWorker.ready.then((registration) => registration.sync.register('titan-sync-queue'));
  }

  window.addEventListener('online', () => flushSyncQueue());
}

export async function flushSyncQueue(limit = 50) {
  await initDB();
  return flushSignals(limit);
}
