@php($manifest = $manifest ?? [])
<div class="alert alert-light border">
    <strong>Doctor manifest</strong>
    <div class="mt-2 small">Workspace: {{ $manifest['workspace'] ?? 'worksuite' }}</div>
</div>
