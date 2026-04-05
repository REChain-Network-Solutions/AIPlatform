@php($menuRisk = $visibilityReport['menu_risk'] ?? [])
<div class="alert {{ ($menuRisk['band'] ?? 'medium') === 'high' ? 'alert-danger' : (($menuRisk['band'] ?? 'medium') === 'low' ? 'alert-success' : 'alert-warning') }} mb-3">
    <strong>Menu visibility risk:</strong> {{ strtoupper($menuRisk['band'] ?? 'medium') }}
    <span class="ml-2">Score: {{ $menuRisk['score'] ?? 'n/a' }}/100</span>
</div>
