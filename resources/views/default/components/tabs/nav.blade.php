@php
    $base_class = 'lqd-tabs-nav inline-flex justify-between gap-1.5 overflow-x-auto rounded-full bg-foreground/[7%] p-1.5 text-xs font-medium';
@endphp

<div {{ $attributes->twMerge($base_class) }}>
    {{ $slot }}
</div>
