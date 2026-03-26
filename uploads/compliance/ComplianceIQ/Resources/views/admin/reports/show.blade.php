@extends('layouts.app')
@section('content')
<h1>{{ $report->title }}</h1>
<p>Period: {{ $report->period_start->toDateString() }} → {{ $report->period_end->toDateString() }}</p>

<form method="post" action="{{ route('admin.compliance.reports.export',$report) }}" class="d-inline">@csrf
  <button class="btn btn-secondary btn-sm">Export CSV</button>
</form>

@if($report->status !== 'signed_off')
<form method="post" action="{{ route('admin.compliance.reports.signoff',$report) }}" class="d-inline">@csrf
  <button class="btn btn-success btn-sm">Sign-off</button>
</form>
@endif

<hr>
<h3>Annotations</h3>
<ul>
  @foreach($report->annotations as $a)
    <li><strong>#{{ $a->id }}</strong> {{ $a->note }} <em>({{ $a->created_at }})</em></li>
  @endforeach
</ul>
<form method="post" action="{{ route('admin.compliance.reports.annotate',$report) }}">
  @csrf
  <textarea class="form-control" name="note" rows="3" placeholder="Add note..."></textarea>
  <button class="btn btn-primary btn-sm mt-2">Add Annotation</button>
</form>
@endsection
