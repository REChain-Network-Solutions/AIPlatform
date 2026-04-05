@php($permissions = $visibility['permissions'] ?? [])
<div class="alert alert-light border">
    <strong>Permission chain</strong>
    <ul class="mt-2 mb-0">
        <li>Status: {{ strtoupper($permissions['status'] ?? 'warn') }}</li>
        <li>{{ $permissions['detail'] ?? 'No permission data' }}</li>
    </ul>
</div>
