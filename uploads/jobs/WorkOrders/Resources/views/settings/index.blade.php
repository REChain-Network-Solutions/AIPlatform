@extends('layouts.app')

@section('content')
<div class="container">
  <h1>Work Orders — Settings</h1>

  @if(session('status'))
    <div class="alert alert-success">{{ session('status') }}</div>
  @endif

  <form method="post" action="{{ route('workorders.settings.update') }}">
    @csrf
    <div class="mb-3">
      <label class="form-label">Require Auth for API</label>
      <input type="hidden" name="api_auth" value="0">
      <input type="checkbox" name="api_auth" value="1" {{ old('api_auth', $cfg['api_auth'] ?? true) ? 'checked' : '' }}>
    </div>

    <div class="mb-3">
      <label class="form-label">Webhook URL</label>
      <input class="form-control" type="url" name="webhook_url" value="{{ old('webhook_url', $cfg['webhook_url'] ?? '') }}">
      <small class="text-muted">Event payloads: WorkOrderCreated, WorkOrderUpdated, WorkOrderCompleted</small>
    </div>

    <div class="mb-3">
      <label class="form-label">Webhook Retries</label>
      <input class="form-control" type="number" name="webhook_retries" min="0" max="10" value="{{ old('webhook_retries', $cfg['webhook_retries'] ?? 3) }}">
    </div>

    <div class="mb-3">
      <label class="form-label">Webhook Backoff (seconds)</label>
      <input class="form-control" type="number" name="webhook_backoff_seconds" min="0" max="120" value="{{ old('webhook_backoff_seconds', $cfg['webhook_backoff_seconds'] ?? 5) }}">
    </div>

    <button class="btn btn-primary" type="submit">Save Settings</button>
  
    <div class="mt-4">
      <form method="post" action="{{ route('workorders.settings.test') }}">
        @csrf
        <button class="btn btn-outline-secondary" type="submit">Test Webhook</button>
      </form>
    </div>
  </form>

</div>
@endsection
