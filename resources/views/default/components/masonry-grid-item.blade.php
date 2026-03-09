<div
    {{ $attributes->merge(['class' => 'masonry-grid-item']) }}
    x-data="{ init: () => $dispatch('masonry:appended', { item: $el }), destroy: () => $dispatch('masonry:remove', { item: $el }) }"
>
    {{ $slot }}
</div>
