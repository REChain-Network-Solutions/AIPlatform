@php($tenant = $visibilityReport['tenant'] ?? [])
<div class="alert alert-light mb-3">
    <strong>Tenant visibility proof</strong>
    @foreach($tenant as $profile => $row)
        <div class="mt-2 small">
            <strong>{{ ucwords(str_replace('_', ' ', $profile)) }}</strong>: {{ !empty($row['visible']) ? 'Visible' : 'Blocked' }}
            @if(!empty($row['proof']['proof'] ?? []))
                <div class="mt-1">{{ collect($row['proof']['proof'])->map(fn($v,$k) => $k . '=' . ($v ? 'yes' : 'no'))->implode(' | ') }}</div>
            @endif
        </div>
    @endforeach
</div>
