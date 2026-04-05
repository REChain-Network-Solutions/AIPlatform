@php($risk = $analysis['risk_profile'] ?? [])
<div class="alert alert-light border">
    <strong>Install profile</strong>
    <ul class="mt-2 mb-0">
        <li>Workspace: {{ $risk['workspace'] ?? 'worksuite' }}</li>
        <li>Repair level: {{ $risk['recommended_repair_level'] ?? 'safe' }}</li>
    </ul>
</div>
