@php
    $wrapper_class = 'lqd-navbar-label-wrap flex w-full min-w-0 items-center gap-2 pb-navbar-link-pb pe-navbar-link-pe ps-navbar-link-ps pt-navbar-link-pt';
    $text_class =
        'lqd-navbar-label inline-block max-w-full min-w-0 flex-1 overflow-hidden text-ellipsis text-4xs uppercase tracking-widest lg:group-[&.navbar-shrinked]/body:w-full lg:group-[&.navbar-shrinked]/body:px-2 lg:group-[&.navbar-shrinked]/body:text-center';
@endphp

<span {{ $attributes->withoutTwMergeClasses()->twMerge($wrapper_class, $attributes->get('class')) }}>
    <span class="{{ $text_class }}">{{ $slot }}</span>
    @if (!empty($badge))
        <x-badge
            class="shrink-0 rounded-md text-[0.5625rem] group-[&.navbar-shrinked]/body:hidden"
            variant="secondary"
        >
            {{ mb_strtoupper($badge) }}
        </x-badge>
    @endif
</span>
