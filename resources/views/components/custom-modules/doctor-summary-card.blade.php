@php($themeClass = ($theme ?? 'primary') === 'success' ? 'module-doctor-card-alt' : '')
<div class="module-doctor-card {{ $themeClass }}">
    <h6>{{ $title }}</h6>
    <p>{{ $body }}</p>
</div>
