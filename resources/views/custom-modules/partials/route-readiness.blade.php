@php($routes = $visibility['routes'] ?? [])
<div class="alert alert-light border">
    <strong>Route readiness</strong>
    <ul class="mt-2 mb-0">
        <li>Status: {{ strtoupper($routes['status'] ?? 'warn') }}</li>
        <li>{{ $routes['detail'] ?? 'No route data' }}</li>
    </ul>
</div>
