@extends('layouts.app')
@section('content')
<h1>Create Compliance Report</h1>
<form method="post" action="{{ route('admin.compliance.reports.store') }}">
  @csrf
  <div class="mb-2"><label>Title</label><input name="title" class="form-control" required></div>
  <div class="mb-2"><label>Period Start</label><input type="date" name="period_start" class="form-control" required></div>
  <div class="mb-2"><label>Period End</label><input type="date" name="period_end" class="form-control" required></div>
  <div class="mb-2"><label>Template</label>
    <select class="form-select" name="filters[template]">
      <option value="baseline">Baseline</option>
      <option value="iso27001">ISO 27001</option>
      <option value="hipaa">HIPAA</option>
    </select>
  </div>
  <button class="btn btn-primary">Create</button>
</form>
@endsection
