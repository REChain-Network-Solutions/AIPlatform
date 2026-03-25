@php
/** Expected: $invoiceId (int) */
@endphp

<div class="card mt-3" id="einvoice-ai-note" data-invoice="{{ $invoiceId }}">
  <div class="card-body">
    <div class="d-flex justify-content-between align-items-center">
      <h5 class="card-title mb-0">AI Invoice Note</h5>
      <button class="btn btn-sm btn-outline-primary" id="einvoice-generate-note">
        Generate AI Note
      </button>
    </div>
    <div class="mt-2 small text-muted" id="einvoice-ai-note-status">Loading latest note…</div>
    <pre class="mt-2 bg-light p-2 small" style="min-height: 60px;" id="einvoice-ai-note-content"></pre>
  </div>
</div>

<script>
(async function() {
  const wrap = document.getElementById('einvoice-ai-note');
  if (!wrap) return;
  const invoiceId = wrap.getAttribute('data-invoice');
  const statusEl = document.getElementById('einvoice-ai-note-status');
  const contentEl = document.getElementById('einvoice-ai-note-content');
  const btn = document.getElementById('einvoice-generate-note');

  async function loadLatest() {
    statusEl.textContent = 'Loading latest note…';
    contentEl.textContent = '';
    const r = await fetch('/einvoice/notes/latest/' + invoiceId, {headers: {'Accept': 'application/json'}});
    if (!r.ok) {
      statusEl.textContent = 'No note found yet.';
      contentEl.textContent = '';
      return;
    }
    const j = await r.json();
    if (j && j.note) {
      statusEl.textContent = 'Last updated: ' + (j.note.updated_at || j.note.created_at || 'n/a');
      contentEl.textContent = j.note.content || '';
    } else {
      statusEl.textContent = 'No note found yet.';
    }
  }

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    statusEl.textContent = 'Queuing generation…';
    try {
      const r = await fetch('/einvoice/ai/generate/' + invoiceId, {
        method: 'POST',
        headers: {'X-CSRF-TOKEN': '{{ csrf_token() }}', 'Accept': 'application/json'}
      });
      const j = await r.json();
      if (j.ok) {
        statusEl.textContent = 'Note generation queued. Refresh this section in a few seconds.';
      } else {
        statusEl.textContent = 'Failed to queue: ' + (j.message || 'Unknown error');
      }
    } catch (e) {
      statusEl.textContent = 'Error: ' + e;
    } finally {
      btn.disabled = false;
    }
  });

  // Initial load
  loadLatest();
  // Optional auto-refresh every 10s for a minute
  let count = 0;
  const iv = setInterval(() => {
    count++;
    if (count > 6) return clearInterval(iv);
    loadLatest();
  }, 10000);
})();
</script>
