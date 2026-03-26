@props(['status' => ''])
@php
    $label = ucfirst(str_replace('_',' ', (string) $status));
@endphp
<span {{ $attributes->merge(['class' => 'badge badge-secondary']) }}>
    {{ $label }}
</span>
