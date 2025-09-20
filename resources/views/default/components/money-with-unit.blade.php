@php
    $base_class = 'flex min-h-7 min-w-7 items-center justify-center rounded-full bg-foreground/[6%] px-2 text-center text-foreground/70 text-sm font-medium';
@endphp
<span {{ $attributes->withoutTwMergeClasses()->twMerge($base_class, $attributes->get('class')) }}>
    @if (is_string($value))
        {{ $value }}
    @else
        {{ $value > 1000 ? floatval($value / 1000) . 'k' : $value }}
    @endif
</span>
