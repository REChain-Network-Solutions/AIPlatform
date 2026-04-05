@php($bridge = $visibility['package_bridge'] ?? [])
<div class="alert alert-light border">
    <strong>Package bridge</strong>
    <ul class="mt-2 mb-0">
        <li>Status: {{ strtoupper($bridge['status'] ?? 'warn') }}</li>
        <li>{{ $bridge['detail'] ?? 'No package bridge data' }}</li>
    </ul>
</div>
