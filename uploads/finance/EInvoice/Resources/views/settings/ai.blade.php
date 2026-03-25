@extends('layouts.app')

@section('content')
<div class="container py-4">
  <h3 class="mb-3">E-Invoice AI Settings</h3>

  <div class="card mb-3">
    <div class="card-body">
      <p class="mb-2"><strong>Environment</strong></p>
      <ul class="mb-0">
        <li>OPENAI_MODEL: <code>{{ config('einvoice.ai.model') }}</code></li>
      </ul>
      <p class="text-muted mb-0">Note: Your API key is read from <code>.env</code> as <code>OPENAI_API_KEY</code>.</p>
    </div>
  </div>

  <div class="card mb-3">
    <div class="card-body">
      <h5 class="card-title">Health Check</h5>
      <button id="ai-health" class="btn btn-outline-primary">Run Health Check</button>
      <pre id="ai-health-out" class="mt-3 small bg-light p-2" style="min-height: 60px;"></pre>
    </div>
  </div>

  <div class="card">
    <div class="card-body">
      <h5 class="card-title">Test Prompt</h5>
      <form id="ai-test-form">
        @csrf
        <div class="mb-2">
          <textarea name="prompt" class="form-control" rows="3" placeholder="Type a short prompt to test the AI..."></textarea>
        </div>
        <button class="btn btn-success">Send Test</button>
      </form>
      <pre id="ai-test-out" class="mt-3 small bg-light p-2" style="min-height: 60px;"></pre>
    </div>
  </div>
</div>

<script>
document.getElementById('ai-health').addEventListener('click', async () => {
  const r = await fetch('{{ route('einvoice.ai.health') }}');
  const j = await r.json();
  document.getElementById('ai-health-out').textContent = JSON.stringify(j, null, 2);
});

document.getElementById('ai-test-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const r = await fetch('{{ route('einvoice.ai.test') }}', {
    method: 'POST',
    headers: {'X-CSRF-TOKEN': '{{ csrf_token() }}'},
    body: fd
  });
  const j = await r.json();
  document.getElementById('ai-test-out').textContent = JSON.stringify(j, null, 2);
});
</script>
@endsection
