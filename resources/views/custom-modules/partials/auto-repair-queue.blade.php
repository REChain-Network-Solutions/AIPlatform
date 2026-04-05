@php($repairQueue = $repairQueue ?? [])
@if(count($repairQueue))
<div class="alert alert-info border">
    <strong>Auto-repair queue</strong>
    <ul class="mt-2 mb-0">
        @foreach($repairQueue as $repair)
            <li><strong>{{ $repair['title'] ?? 'Repair' }}:</strong> {{ $repair['detail'] ?? '' }} <span class="badge badge-light border">{{ strtoupper($repair['mode'] ?? 'safe') }}</span></li>
        @endforeach
    </ul>
</div>
@endif
