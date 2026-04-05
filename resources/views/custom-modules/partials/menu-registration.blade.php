@php($menuRegistration = $visibilityReport['menu_registration'] ?? [])
<div class="alert {{ ($menuRegistration['status'] ?? 'warn') === 'pass' ? 'alert-success' : 'alert-warning' }} mb-3">
    <strong>Live menu registration:</strong> {{ $menuRegistration['detail'] ?? 'No menu registration data.' }}
    @if(!empty($menuRegistration['hits'] ?? []))
        <div class="mt-2 small">{{ implode(', ', $menuRegistration['hits']) }}</div>
    @endif
</div>
