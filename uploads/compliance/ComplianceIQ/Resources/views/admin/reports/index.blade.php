@extends('layouts.app')
@section('content')
<h1>Compliance Reports</h1>
<form method="get" class="mb-3">
  <select name="status"><option value="">All</option><option>draft</option><option>in_review</option><option>signed_off</option></select>
  <button class="btn btn-primary btn-sm">Filter</button>
</form>
<table class="table">
  <thead><tr><th>Title</th><th>Period</th><th>Status</th><th></th></tr></thead>
  <tbody>
  @foreach ($reports as $r)
    <tr>
      <td>{{ $r->title }}</td>
      <td>{{ $r->period_start->toDateString() }} → {{ $r->period_end->toDateString() }}</td>
      <td>{{ $r->status }}</td>
      <td>
        <a href="{{ route('admin.compliance.reports.show',$r) }}" class="btn btn-link btn-sm">Open</a>
      </td>
    </tr>
  @endforeach
  </tbody>
</table>
{{ $reports->links() }}
@endsection
