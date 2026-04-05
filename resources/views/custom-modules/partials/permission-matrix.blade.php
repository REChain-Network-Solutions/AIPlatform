@php($matrix = $visibilityReport['permissions']['role_matrix']['matrix'] ?? [])
<div class="alert {{ empty($matrix) ? 'alert-warning' : 'alert-success' }} mb-3">
    <strong>Permission role matrix:</strong>
    {{ empty($matrix) ? 'No role mappings detected for inferred permission keys.' : 'Role mappings detected for inferred permission keys.' }}
    @if(!empty($matrix))
        <ul class="mt-2 mb-0 small">
            @foreach($matrix as $permission => $roles)
                <li><strong>{{ $permission }}</strong> → role IDs: {{ implode(', ', $roles) }}</li>
            @endforeach
        </ul>
    @endif
</div>
