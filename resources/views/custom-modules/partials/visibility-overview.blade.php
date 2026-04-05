@php($visibility = $visibility ?? [])
@if(!empty($visibility))
<div class="alert {{ ($visibility['final']['will_appear'] ?? false) ? 'alert-success' : 'alert-warning' }} border">
    <strong>User menu visibility</strong>
    <div class="mt-2">{{ ($visibility['final']['detail'] ?? 'Visibility not evaluated.') }}</div>
</div>
@endif
