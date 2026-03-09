@php
    $classname = @twMerge('lqd-shimmer-text bg-clip-text bg-gradient-to-r from-foreground via-background to-foreground text-transparent leading-[1.2em]', $attributes->get('class'));
@endphp

<span class="{{ $classname }}">
    {{ $slot }}
</span>
