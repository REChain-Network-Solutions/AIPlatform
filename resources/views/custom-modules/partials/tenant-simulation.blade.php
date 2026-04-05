@php($tenant = $visibility['tenant'] ?? [])
<div class="alert alert-light border">
    <strong>Tenant simulation</strong>
    <ul class="mt-2 mb-0">
        @foreach($tenant as $profile => $result)
            <li><strong>{{ str_replace('_', ' ', $profile) }}:</strong> {{ strtoupper($result['status'] ?? 'warn') }} — {{ $result['detail'] ?? '' }}</li>
        @endforeach
    </ul>
</div>
