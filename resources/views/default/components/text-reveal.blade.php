@props([
    'start' => 'top bottom',
    'end' => 'center 65%',
    'animateFrom' => [
        'opacity' => 0.2,
    ],
    'animateTo' => [
        'opacity' => 1,
    ],
])

<span
    class="lqd-text-reveal-el"
    x-data='liquidTextReveal({ start: @json($start), end: @json($end), animateFrom: @json($animateFrom), animateTo: @json($animateTo) })'
>
    {{ $slot }}
</span>
