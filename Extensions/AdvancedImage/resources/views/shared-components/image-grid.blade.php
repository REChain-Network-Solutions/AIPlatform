<template x-for="item in newItems">
    <div
        class="lqd-adv-editor-recent-images-grid-item image-result group/item"
        data-id-prefix="{{ $id_prefix ?? 'modal-' }}"
        :data-id="item.id"
        :data-generator="item.response.toLowerCase()"
        :data-payload="JSON.stringify(item)"
        :data-request-id="item.request_id ? item.request_id : ''"
    >
        <div
            class="relative mb-4 grid aspect-[1/0.85] w-full place-items-center overflow-hidden rounded-lg bg-heading-foreground/5 transition-all group-hover/item:shadow-[0_38px_55px_rgba(0,0,0,0.1)]">
            <img
                class="size-full object-cover object-center transition-all group-hover/item:scale-105"
                :id="`img-${item.response}-${item.id}`"
                :src="item.output"
                :alt="item.title"
            >

            <div class="absolute inset-0 flex items-center justify-center gap-1 opacity-0 transition-opacity group-hover/item:opacity-100">
                <x-button
                    class="lqd-image-result-view gallery size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white focus:bg-background focus:bg-emerald-400 focus:text-white"
                    ::id="`img-${item.response}-${item.id}-payload`"
                    @click.prevent="setActiveModal( JSON.parse($el.closest('.lqd-adv-editor-recent-images-grid-item').getAttribute('data-payload') || {}), '{{ $id_prefix ?? 'modal-' }}' ); modalShow = true"
                    size="none"
                    href="#"
                >
                    <x-tabler-eye class="size-5" />
                </x-button>
                <x-button
                    class="lqd-image-result-edit delete size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white focus:bg-background focus:bg-emerald-400 focus:text-white"
                    size="none"
                    @click.prevent="editingImage = JSON.parse($el.closest('.lqd-adv-editor-recent-images-grid-item').getAttribute('data-payload') || {}); selectedTool = 'uncrop'; switchView( 'editor' );"
                >
                    <x-tabler-pencil-minus class="size-5" />
                </x-button>
                <x-button
                    class="lqd-image-result-download download size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white focus:bg-background focus:bg-emerald-400 focus:text-white"
                    ::id="`img-${item.response}-${item.id}-download`"
                    size="none"
                    ::href="item.output"
                    ::download="item.slug"
                >
                    <x-tabler-circle-chevron-down class="size-5" />
                </x-button>
            </div>
        </div>
        <h5 class="mb-0 hyphens-auto break-all text-2xs font-medium">
            <span x-text="item.title"></span>
        </h5>
        <time class="text-2xs font-medium opacity-60">
            <span x-text="new Date(item.created_at).toLocaleDateString()"></span>
        </time>
    </div>
</template>
@foreach ($images as $item)
    <div
        class="lqd-adv-editor-recent-images-grid-item image-result group/item"
        data-id="{{ $item->id }}"
        data-id-prefix="{{ $id_prefix ?? 'modal-' }}"
        data-generator="{{ str()->lower($item->response) }}"
        data-payload="{{ json_encode($item) }}"
        {{ $item->request_id ? 'data-request-id=' . $item->request_id : '' }}
    >
        <div
            class="relative mb-4 grid aspect-[1/0.85] w-full place-items-center overflow-hidden rounded-lg bg-heading-foreground/5 transition-all group-hover/item:shadow-[0_38px_55px_rgba(0,0,0,0.1)]">
            @if ($item->output_url)
                <img
                    class="size-full object-cover object-center transition-all group-hover/item:scale-105"
                    id="img-{{ $item->response }}-{{ $item->id }}"
                    src="{{ ThumbImage($item->output_url) }}"
                    alt="{{ $item->prompt ? \Illuminate\Support\Str::limit($item->prompt, 40) : $item->title }}"
                >
                <div class="absolute inset-0 flex items-center justify-center gap-1 opacity-0 transition-opacity group-hover/item:opacity-100">
                    <x-button
                        class="lqd-image-result-view gallery size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white focus:bg-background focus:bg-emerald-400 focus:text-white"
                        id="img-{{ $item->response . '-' . $item->id }}-payload"
                        @click.prevent="setActiveModal( JSON.parse($el.closest('.lqd-adv-editor-recent-images-grid-item').getAttribute('data-payload') || {}), '{{ $id_prefix ?? 'modal-' }}' ); modalShow = true"
                        size="none"
                        href="#"
                    >
                        <x-tabler-eye class="size-5" />
                    </x-button>
                    <x-button
                        class="lqd-image-result-edit delete size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white focus:bg-background focus:bg-emerald-400 focus:text-white"
                        size="none"
                        @click.prevent="editingImage = JSON.parse($el.closest('.lqd-adv-editor-recent-images-grid-item').getAttribute('data-payload') || {}); selectedTool = 'uncrop'; switchView( 'editor' );"
                    >
                        <x-tabler-pencil-minus class="size-5" />
                    </x-button>
                    <x-button
                        class="lqd-image-result-download download size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white focus:bg-background focus:bg-emerald-400 focus:text-white"
                        id="img-{{ $item->response }}-{{ $item->id }}-download"
                        size="none"
                        href="{{ $item->output_url }}"
                        download="{{ $item->slug }}"
                    >
                        <x-tabler-circle-chevron-down class="size-5" />
                    </x-button>
                </div>
            @endif
        </div>
        <h5 class="mb-0 hyphens-auto break-all text-2xs font-medium">
            {{ $item->title }}
        </h5>
        <time class="text-2xs font-medium opacity-60">
            {{ \Carbon\Carbon::parse($item->created_at)->diffForHumans() }}
        </time>
    </div>
@endforeach
