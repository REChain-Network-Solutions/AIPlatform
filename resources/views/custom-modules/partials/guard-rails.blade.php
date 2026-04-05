@php($guardRails = $visibilityReport['guard_rails'] ?? [])
@if(!empty($guardRails['items'] ?? []))
    <div class="alert alert-warning mb-3">
        <strong>Install guard rails</strong>
        <ul class="mb-0 mt-2 small">
            @foreach($guardRails['items'] as $item)
                <li>{{ $item }}</li>
            @endforeach
        </ul>
    </div>
@endif
