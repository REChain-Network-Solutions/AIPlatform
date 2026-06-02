@aware(['default' => null, 'insideContents' => false])

@php
    $base_class = $insideContents
        ? 'lqd-tabs-content col-start-1 col-end-1 row-start-1 row-end-1 w-full min-w-0'
        : 'lqd-tabs-content col-start-1 col-end-1 row-start-2 row-end-2 w-full min-w-0';
    $isActive = $name === $default;
@endphp

<div
    {{ $attributes->twMerge($base_class) }}
    @if (!$isActive) x-cloak @endif
    x-show="activeTab === {{ json_encode($name) }}"
    x-transition:enter="transition"
    x-transition:enter-start="opacity-0 -translate-x-1"
    x-transition:enter-end="opacity-100 translate-x-0"
    x-transition:leave="transition"
    x-transition:leave-start="opacity-100 translate-x-0"
    x-transition:leave-end="opacity-0 translate-x-1"
>
    {{ $slot }}
</div>
