@php
    $base_class = 'lqd-tabs-contents grid w-full min-w-0 grid-cols-1 place-items-start';
@endphp

<div {{ $attributes->twMerge($base_class) }}>
    {{ $slot }}
</div>
