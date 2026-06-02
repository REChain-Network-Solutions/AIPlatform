@php
    $base_class = 'lqd-tabs grid gap-8 grid-cols-1 [grid-template-rows:auto_1fr]';
@endphp

<div
    {{ $attributes->twMerge($base_class) }}
    x-data="liquidTabs({ activeTab: {{ json_encode($default) }} })"
>
    {{ $slot }}
</div>
