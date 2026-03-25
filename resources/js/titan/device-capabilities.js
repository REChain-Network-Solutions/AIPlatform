export function detectCapabilities() {
  return {
    webGPU: typeof navigator !== 'undefined' && 'gpu' in navigator,
    webAssembly: typeof WebAssembly !== 'undefined',
    speechRecognition:
      typeof window !== 'undefined' &&
      ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window),
    localLlm: typeof localStorage !== 'undefined' && !!localStorage.getItem('titan_llm_endpoint'),
    serviceWorker: typeof navigator !== 'undefined' && 'serviceWorker' in navigator,
    indexedDB: typeof indexedDB !== 'undefined',
  };
}

export function getExecutionTier() {
  const caps = detectCapabilities();
  if (caps.webGPU || caps.localLlm) return 'device';
  if (caps.serviceWorker && caps.indexedDB) return 'local';
  return 'cloud';
}
