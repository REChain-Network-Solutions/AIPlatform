@props(['href' => '#', 'label' => 'Open'])
<a {{ $attributes->merge(['class' => 'btn btn-outline-primary']) }} href="{{ $href }}">{{ $label }}</a>
