@php($sidebar = $visibility['sidebar'] ?? [])
<div class="alert alert-light border">
    <strong>Sidebar readiness</strong>
    <ul class="mt-2 mb-0">
        <li>Status: {{ strtoupper($sidebar['status'] ?? 'warn') }}</li>
        <li>{{ $sidebar['detail'] ?? 'No sidebar data' }}</li>
    </ul>
</div>
