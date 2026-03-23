@props([
    'itemSelector' => '.masonry-grid-item',
])

<div
    {{ $attributes->merge(['class' => 'masonry-grid']) }}
    x-masonry="{
        itemSelector: '{{ $itemSelector }}',
        percentPosition: true,
        masonry: {
            columnWidth: '{{ $itemSelector }}',
        },
		transitionDuration: 100
    }"
>
    {{ $slot }}
</div>

@pushOnce('script')
    <script src="{{ custom_theme_url('/assets/libs/isotope/isotope.pkgd.min.js') }}"></script>
    <script src="{{ custom_theme_url('/assets/libs/imagesloaded/imagesloaded.pkgd.min.js') }}"></script>
@endPushOnce
