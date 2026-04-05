@php($narrative = $visibilityReport['narrative'] ?? [])
<div class="alert alert-info mb-3">
    <strong>Doctor narrative:</strong> {{ $narrative['summary'] ?? 'No narrative available.' }}
</div>
