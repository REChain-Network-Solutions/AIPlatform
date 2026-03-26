<!— created —>
@extends('feedback::layouts.master')
@section('content')
<h2>CSAT Surveys</h2>
<form method="post" action="{{ route('feedback.csat.store') }}" class="mb-4">
  @csrf
  <div class="row g-2">
    <div class="col-md-6"><input class="form-control" name="title" placeholder="Survey title" required></div>
    <div class="col-md-6 d-flex align-items-center"><button class="btn btn-primary">Create CSAT Survey</button></div>
  </div>
</form>
<table class="table"><thead><tr><th>Survey</th><th style="width:50%">Distribution</th><th>Total</th><th>Avg</th><th>Public Link</th></tr></thead><tbody>
@foreach($surveys as $s)
@php $a = $agg[$s->id] ?? null; $t = max(1,(int)($a->total ?? 0)); $avg = round(((5*($a->r5??0)+4*($a->r4??0)+3*($a->r3??0)+2*($a->r2??0)+1*($a->r1??0))/$t),2);
$parts=[5=>(int)($a->r5??0),4=>(int)($a->r4??0),3=>(int)($a->r3??0),2=>(int)($a->r2??0),1=>(int)($a->r1??0)]; $total=array_sum($parts); @endphp
<tr><td>{{ $s->title }}</td>
<td><div class="d-flex" style="height:18px; background:#f2f2f2;">
@foreach([5,4,3,2,1] as $r) @php $w = $total ? round(($parts[$r]/$total)*100) : 0; @endphp
<div title="{{ $r }}★: {{ $parts[$r] }}" style="width: {{ $w }}%; border-right:1px solid #fff;"></div>
@endforeach
</div>
<small>5★ {{ $parts[5]??0 }} · 4★ {{ $parts[4]??0 }} · 3★ {{ $parts[3]??0 }} · 2★ {{ $parts[2]??0 }} · 1★ {{ $parts[1]??0 }}</small>
</td><td>{{ $total }}</td><td>{{ $avg }}</td>
<td><code>{{ url('/feedback/survey/csat/'.$s->id) }}</code></td></tr>
@endforeach
</tbody></table>
@endsection
