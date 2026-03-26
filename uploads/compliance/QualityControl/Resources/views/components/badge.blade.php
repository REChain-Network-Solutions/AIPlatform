@props(['text' => '', 'class' => ''])
<span {{ $attributes->merge(['class' => 'badge badge-light ' . $class]) }}>
    {{ $text }}
</span>
