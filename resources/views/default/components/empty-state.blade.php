@props([
    'icon' => 'tabler-mood-empty',
    'title' => '',
    'description' => '',
    'show' => null,
])

@php
    $base_class = 'lqd-empty-state text-center';
@endphp

<div
    {{ $attributes->withoutTwMergeClasses()->twMerge($base_class) }}
    @if ($show) x-show="{{ $show }}" @endif
>
    <div class="mx-auto mb-3 grid size-28 place-items-center rounded-full bg-foreground/[3%]">
        <x-dynamic-component
            class="size-10"
            :component="$icon"
        />
    </div>
    @if ($title)
        <p class="mb-1 text-lg font-semibold text-heading-foreground">
            {{ $title }}
        </p>
    @endif
    @if ($description)
        <p class="m-0 mx-auto max-w-80 text-pretty text-xs text-foreground/60">
            {{ $description }}
        </p>
    @endif
</div>
