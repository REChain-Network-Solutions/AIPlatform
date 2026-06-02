@aware(['default' => null])

@php
    $base_class =
        'lqd-tabs-nav-trigger grow whitespace-nowrap rounded-full px-6 py-2.5 text-xs font-medium leading-tight transition-all hover:bg-background/90 [&.active]:bg-background [&.active]:shadow-[0_2px_13px_hsl(0_0%_0%/10%)]';
    $isActive = $name === $default;
@endphp

<button
    type="button"
    {{ $attributes->twMerge($base_class, $isActive ? 'active' : '') }}
    :class="{ 'active': activeTab === {{ json_encode($name) }} }"
    @click.prevent="activeTab = {{ json_encode($name) }}"
>
    {{ $slot }}
</button>
