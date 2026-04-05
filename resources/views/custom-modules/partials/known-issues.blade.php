@php($issues = $visibility['known_issues'] ?? [])
@if(count($issues))
<div class="alert alert-warning border">
    <strong>Known visibility issues</strong>
    <ul class="mt-2 mb-0">
        @foreach($issues as $issue)
            <li><strong>{{ $issue['title'] ?? 'Issue' }}:</strong> {{ $issue['detail'] ?? '' }}</li>
        @endforeach
    </ul>
</div>
@endif
