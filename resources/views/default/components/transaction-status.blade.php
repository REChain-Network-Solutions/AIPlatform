@php
    $base_class = 'inline-flex items-center leading-none text-base font-semibold leading-snug rounded-md';
    $color = $status == 'Pending' ? '#B58500' : ($status == 'Completed' ? '#118C60' : 'var(--foreground)');
@endphp

<span {{ $attributes->withoutTwMergeClasses()->twMerge($base_class, $attributes->get('class')) }}
    style="color: {{ $color }}">
    {{ $status }}
</span>
