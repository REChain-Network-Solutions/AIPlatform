@php($nextSteps = $nextSteps ?? [])
@if(count($nextSteps))
<div class="alert alert-light border">
    <strong>Checklist</strong>
    <ol class="mt-2 mb-0">
        @foreach($nextSteps as $step)
            <li>{{ $step }}</li>
        @endforeach
    </ol>
</div>
@endif
