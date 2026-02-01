<template
    x-for="image in newImages"
    :key="'{{ $id_prefix ?? 'modal-' }}' + image.id"
>
    <div
        class="lqd-realtime-image-recent-images-grid-item lqd-realtime-image-new image-result group/item"
        data-id-prefix="{{ $id_prefix ?? 'modal-' }}"
        :data-id="image.id"
        :data-payload="JSON.stringify(image)"
    >
        <div
            class="relative mb-4 grid aspect-[1/0.85] w-full place-items-center overflow-hidden rounded-lg bg-heading-foreground/5 transition-all group-hover/item:shadow-[0_38px_55px_rgba(0,0,0,0.1)]">
            <img
                class="size-full object-cover object-center transition-all group-hover/item:scale-105"
                :src="image.image"
                :alt="image.prompt"
                loading="lazy"
            >

            <div class="absolute inset-0 flex items-center justify-center gap-1 opacity-0 transition-opacity group-hover/item:opacity-100">
                <x-button
                    class="lqd-image-result-view gallery size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white focus:bg-background focus:bg-emerald-400 focus:text-white"
                    @click.prevent="setActiveModal( JSON.parse($el.closest('.lqd-realtime-image-recent-images-grid-item').getAttribute('data-payload') || {}), '{{ $id_prefix ?? 'modal-' }}' ); modalShow = true"
                    size="none"
                    href="#"
                >
                    <x-tabler-eye class="size-5" />
                </x-button>
                <x-button
                    class="lqd-image-result-download download size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white focus:bg-background focus:bg-emerald-400 focus:text-white"
                    size="none"
                    ::href="image.image"
                    ::download="image.prompt"
                >
                    <x-tabler-circle-chevron-down class="size-5" />
                </x-button>
            </div>
        </div>
        <h5 class="mb-0 hyphens-auto break-all text-2xs font-medium">
            <span x-text="image.prompt"></span>
        </h5>
        <time class="text-2xs font-medium opacity-60">
            <span x-text="image.formatted_date || new Date(image.created_at).toLocaleDateString()"></span>
        </time>
    </div>
</template>

@foreach ($images as $image)
    <div
        class="lqd-realtime-image-recent-images-grid-item image-result group/item"
        data-id="{{ $image->id }}"
        data-id-prefix="{{ $id_prefix ?? 'modal-' }}"
        data-payload="{{ json_encode($image) }}"
    >
        <div
            class="relative mb-4 grid aspect-[1/0.85] w-full place-items-center overflow-hidden rounded-lg bg-heading-foreground/5 transition-all group-hover/item:shadow-[0_38px_55px_rgba(0,0,0,0.1)]">
            @if ($image->image)
                <img
                    class="size-full object-cover object-center transition-all group-hover/item:scale-105"
                    id="{{ $image->response['id'] }}"
                    src="{{ ThumbImage($image->image) }}"
                    alt="{{ str()->limit($image->prompt, 40) }}"
                    loading="lazy"
                >
                <div class="absolute inset-0 flex items-center justify-center gap-1 opacity-0 transition-opacity group-hover/item:opacity-100">
                    <x-button
                        class="lqd-image-result-view gallery size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white focus:bg-background focus:bg-emerald-400 focus:text-white"
                        @click.prevent="setActiveModal( JSON.parse($el.closest('.lqd-realtime-image-recent-images-grid-item').getAttribute('data-payload') || {}), '{{ $id_prefix ?? 'modal-' }}' ); modalShow = true"
                        size="none"
                        href="#"
                    >
                        <x-tabler-eye class="size-5" />
                    </x-button>
                    <x-button
                        class="lqd-image-result-download download size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white focus:bg-background focus:bg-emerald-400 focus:text-white"
                        size="none"
                        href="{{ $image->image }}"
                        download="{{ $image->prompt }}"
                    >
                        <x-tabler-circle-chevron-down class="size-5" />
                    </x-button>
                </div>
            @endif
        </div>
        <h5 class="mb-0 hyphens-auto break-all text-2xs font-medium">
            {{ $image->prompt }}
        </h5>
        <time class="text-2xs font-medium opacity-60">
            {{ $image->created_at->diffInMinutes() < 1 ? __('Just now') : $image->created_at->diffForHumans() }}
        </time>
    </div>
@endforeach
