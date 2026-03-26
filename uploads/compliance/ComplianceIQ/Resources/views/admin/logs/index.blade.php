@extends('layouts.app')
@section('content')
<h1>Compliance Logs</h1>
<form method="get" class="mb-3">
  <input name="type" placeholder="hashable type" class="form-control mb-2" value="{{ request('type') }}">
  <select name="status" class="form-select mb-2">
    <option value="">Any status</option>
    <option value="valid" @selected(request('status')==='valid')>valid</option>
    <option value="mismatch" @selected(request('status')==='mismatch')>mismatch</option>
    <option value="unknown" @selected(request('status')==='unknown')>unknown</option>
  </select>
  <button class="btn btn-primary btn-sm">Filter</button>
</form>
<table class="table">
  <thead><tr><th>ID</th><th>Type</th><th>Ref</th><th>SHA256</th><th>Status</th><th>Computed</th></tr></thead>
  <tbody>
  @foreach($logs as $l)
    <tr>
      <td>{{ $l->id }}</td>
      <td>{{ $l->hashable_type }}</td>
      <td>{{ $l->hashable_id }}</td>
      <td><code>{{ $l->sha256 }}</code></td>
      <td>{{ $l->status }}</td>
      <td>{{ $l->computed_at }}</td>
    </tr>
  @endforeach
  </tbody>
</table>
{{ $logs->links() }}
@endsection
