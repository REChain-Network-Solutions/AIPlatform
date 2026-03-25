@extends('layouts.app')
@section('content')
<div class="container">
  <h3>FSM Dashboard</h3>
  <div class="row g-3 mb-3">
    <div class="col-md-3"><div class="card"><div class="card-body"><div class="small text-muted">Open @term('work_order')s</div><div class="display-6">{{ $stats['wo_open'] }}</div></div></div></div>
    <div class="col-md-3"><div class="card"><div class="card-body"><div class="small text-muted">New Today</div><div class="display-6">{{ $stats['wo_today'] }}</div></div></div></div>
    <div class="col-md-3"><div class="card"><div class="card-body"><div class="small text-muted">SLA Breaches</div><div class="display-6">{{ $stats['sla_breach'] }}</div></div></div></div>
  </div>
  <h5>Dispatch</h5>
  <div class="row g-3 mb-4">
    <div class="col-md-3"><div class="card"><div class="card-body"><div class="small text-muted">Assignments Today</div><div class="h3">{{ $dispatch['today_assignments'] }}</div></div></div></div>
    <div class="col-md-3"><div class="card"><div class="card-body"><div class="small text-muted">Unscheduled</div><div class="h3">{{ $dispatch['unassigned'] }}</div></div></div></div>
  </div>
</div>
@endsection