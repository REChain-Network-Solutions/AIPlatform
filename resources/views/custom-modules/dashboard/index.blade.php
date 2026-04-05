@extends('layouts.app')
@section('content')
<div class="w-100 d-flex ">@include('sections.setting-sidebar')
<x-setting-card>
<x-slot name="header"><div class="s-b-n-header"><h2 class="mb-0 p-20 f-21 font-weight-normal border-bottom-grey">Module Installer Dashboard</h2></div></x-slot>
<div class="p-4">
<div class="row mb-4">
@foreach([
 ['label'=>'Total Installations','value'=>$stats['total_installations'] ?? 0],
 ['label'=>'Success Rate','value'=>($stats['success_rate'] ?? 0).'%'],
 ['label'=>'Rollback Ready','value'=>$stats['rollback_available'] ?? 0],
 ['label'=>'Test Pass Rate','value'=>($testSummary['success_rate'] ?? 0).'%'],
] as $card)
<div class="col-md-3 mb-3"><div class="card border h-100"><div class="card-body"><div class="text-muted small">{{ $card['label'] }}</div><div class="h3 mb-0">{{ $card['value'] }}</div></div></div></div>
@endforeach
</div>
<div class="row mb-4">
<div class="col-md-3 mb-3"><div class="card border h-100"><div class="card-header">Dependency Summary</div><div class="card-body"><div class="small text-muted">Total relationships</div><div class="h4">{{ $dependencySummary['total'] ?? 0 }}</div><div class="small mt-2">Resolved: {{ $dependencySummary['resolved'] ?? 0 }} · Pending: {{ $dependencySummary['pending'] ?? 0 }}</div></div></div></div>
<div class="col-md-3 mb-3"><div class="card border h-100"><div class="card-header">Automated Tests</div><div class="card-body"><div class="small text-muted">Executed checks</div><div class="h4">{{ $testSummary['total'] ?? 0 }}</div><div class="small mt-2">Passed: {{ $testSummary['passed'] ?? 0 }} · Failed: {{ $testSummary['failed'] ?? 0 }}</div></div></div></div>
<div class="col-md-3 mb-3"><div class="card border h-100"><div class="card-header">Registry Feeds</div><div class="card-body"><div class="small text-muted">Active feeds / indexed items</div><div class="h4">{{ $registrySummary['active_feeds'] ?? 0 }} / {{ $registrySummary['items'] ?? 0 }}</div><div class="small mt-2">Total feeds: {{ $registrySummary['feeds'] ?? 0 }} · Stale items: {{ $registrySummary['stale_items'] ?? 0 }}</div></div></div></div>
<div class="col-md-3 mb-3"><div class="card border h-100"><div class="card-header">Runtime Policy</div><div class="card-body"><ul class="list-unstyled mb-0 small"><li>Security validator: {{ ($configurationSummary['security_validator'] ?? false) ? 'On' : 'Off' }}</li><li>Permission validator: {{ ($configurationSummary['permission_validator'] ?? false) ? 'On' : 'Off' }}</li><li>Route validator: {{ ($configurationSummary['route_validator'] ?? false) ? 'On' : 'Off' }}</li><li>Cache clear repair: {{ ($configurationSummary['cache_clear'] ?? false) ? 'On' : 'Off' }}</li></ul></div></div></div>
</div>
<div class="row">
<div class="col-md-6 mb-4"><div class="card border h-100"><div class="card-header">Recent Installations</div><div class="card-body p-0"><div class="table-responsive"><table class="table mb-0"><thead><tr><th>Module</th><th>Status</th><th>Installed</th></tr></thead><tbody>@forelse($recentInstallations as $install)<tr><td><a href="{{ route('modules.show',$install->id) }}">{{ $install->module_name }}</a></td><td>{{ $install->status }}</td><td>{{ optional($install->installed_at)->diffForHumans() }}</td></tr>@empty<tr><td colspan="3" class="text-muted">No installations yet.</td></tr>@endforelse</tbody></table></div></div></div></div>
<div class="col-md-6 mb-4"><div class="card border h-100"><div class="card-header">Slowest Installations</div><div class="card-body p-0"><div class="table-responsive"><table class="table mb-0"><thead><tr><th>Module</th><th>Duration</th></tr></thead><tbody>@forelse(($performanceMetrics['slowest_installations'] ?? []) as $metric)<tr><td>{{ $metric['module'] }}</td><td>{{ $metric['duration'] }}s</td></tr>@empty<tr><td colspan="2" class="text-muted">No performance data yet.</td></tr>@endforelse</tbody></table></div></div></div></div>
</div>
</div>
</x-setting-card></div>
@endsection
