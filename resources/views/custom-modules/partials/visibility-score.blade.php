@php($final = $visibility['final'] ?? [])
<div class="alert {{ ($final['status'] ?? 'warn') === 'pass' ? 'alert-success' : (($final['status'] ?? 'warn') === 'fail' ? 'alert-danger' : 'alert-warning') }} border">
    <strong>Visibility score</strong>
    <div class="mt-2">{{ $final['detail'] ?? 'No score' }}</div>
    <div class="small mt-1">Warnings: {{ $final['warning_count'] ?? 0 }}</div>
</div>
