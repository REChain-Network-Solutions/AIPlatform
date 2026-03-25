@php
/** Optional: $clientId (int) */
@endphp

<div class="card mt-3" id="einvoice-ai-create" data-client="{{ $clientId ?? '' }}">
  <div class="card-body">
    <h5 class="card-title">Use AI to Create Invoice</h5>
    <form id="einvoice-ai-create-form">
      @csrf
      <div class="mb-2">
        <label class="form-label">Describe what to bill</label>
        <textarea name="prompt" class="form-control" rows="3" placeholder="Example: 3 hours plumbing @ $120/hr, 1x water filter $80, due in 14 days"></textarea>
      </div>
      @if(!empty($clientId))
      <input type="hidden" name="client_id" value="{{ (int)$clientId }}">
      @endif
      <div class="d-flex gap-2">
        <button class="btn btn-primary" id="einvoice-ai-draft-btn">Create Draft</button>
        <button type="button" class="btn btn-outline-secondary" id="einvoice-ai-create-btn" disabled>Create Invoice</button>
      </div>
    </form>

    <div class="mt-3">
      <div class="small text-muted">Draft preview:</div>
      <pre id="einvoice-ai-draft-out" class="bg-light p-2 small" style="min-height: 80px;"></pre>
    </div>
  </div>
</div>

<script>
(function(){
  const form = document.getElementById('einvoice-ai-create-form');
  const draftOut = document.getElementById('einvoice-ai-draft-out');
  const createBtn = document.getElementById('einvoice-ai-create-btn');
  let lastDraftId = null;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    createBtn.disabled = true;
    draftOut.textContent = 'Drafting...';

    const fd = new FormData(form);
    const r = await fetch('{{ route('einvoice.ai.draft') }}', {
      method: 'POST',
      headers: {'X-CSRF-TOKEN': '{{ csrf_token() }}', 'Accept': 'application/json'},
      body: fd
    });
    const j = await r.json();
    if (j.ok) {
      lastDraftId = j.draft_id;
      draftOut.textContent = JSON.stringify(j.draft, null, 2);
      createBtn.disabled = false;
    } else {
      draftOut.textContent = 'Error: ' + (j.error || 'Unknown') + '\nRaw: ' + (j.raw || '');
    }
  });

  createBtn.addEventListener('click', async () => {
    if (!lastDraftId) return;
    createBtn.disabled = true;
    const r = await fetch('/einvoice/ai/create-from-draft/' + lastDraftId, {
      method: 'POST',
      headers: {'X-CSRF-TOKEN': '{{ csrf_token() }}', 'Accept': 'application/json'}
    });
    const j = await r.json();
    if (j.ok) {
      draftOut.textContent = 'Invoice created with ID: ' + j.invoice_id;
    } else {
      draftOut.textContent = 'Could not auto-create invoice. Here is your draft to copy into the native form:\n' + JSON.stringify(j.draft, null, 2);
    }
    createBtn.disabled = false;
  });
})();
</script>
