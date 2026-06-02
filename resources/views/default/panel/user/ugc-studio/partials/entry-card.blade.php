{{-- One Studio video card; rendered inside an x-for binding `entry`. --}}
<x-card
    class="lqd-video-card overflow-hidden transition hover:-translate-y-0.5 hover:shadow-xl hover:shadow-black/5 [&.selected]:ring-2 [&.selected]:ring-primary"
    ::class="{ 'selected': isSelected(entry) }"
    size="none"
>
    <figure class="relative aspect-[1/0.7] overflow-hidden bg-foreground/5">
        <template x-if="entry.status === 'completed' && entry.video_url">
            <video
                class="size-full object-contain"
                :src="entry.video_url + '#t=0.1'"
                preload="metadata"
                muted
                playsinline
            ></video>
        </template>
        <template x-if="entry.status === 'completed' && !entry.video_url && entry.thumb_url">
            <img
                class="size-full object-contain"
                :src="entry.thumb_url"
                alt=""
            >
        </template>
        <template x-if="entry.status !== 'completed' && entry.status !== 'failed'">
            <div class="inline-grid size-full cursor-default place-items-center bg-background/5 text-center">
                <x-shimmer-text>
                    {{ __('Processing...') }}
                </x-shimmer-text>
            </div>
        </template>
        <template x-if="entry.status === 'failed'">
            <div
                class="inline-grid size-full cursor-help place-items-center bg-background/5 px-2 text-center"
                :title="entry.error || '{{ __('Video generation failed.') }}'"
            >
                <div>
                    <p class="mb-1 text-xs font-medium">{{ __('Failed to Generate') }}</p>
                    <p
                        class="m-0 line-clamp-2 text-3xs opacity-60"
                        x-show="entry.error"
                        x-text="entry.error"
                    ></p>
                </div>
            </div>
        </template>

        {{-- Selection checkbox --}}
        <label
            class="absolute end-2 top-2 z-3 inline-grid size-6 cursor-pointer place-items-center rounded border bg-white/90 opacity-0 shadow-sm shadow-black/5 backdrop-blur-sm transition group-hover/card:opacity-100 group-[&.selected]/card:bg-primary group-[&.selected]/card:text-primary-foreground group-[&.selected]/card:opacity-100"
        >
            <input
                class="hidden"
                type="checkbox"
                :checked="isSelected(entry)"
                @change="toggleEntrySelection(entry)"
            />
            <x-tabler-check class="size-3.5 opacity-0 transition group-[&.selected]/card:opacity-100" />
        </label>
    </figure>
    <div class="p-3">
        <div class="flex items-center gap-2">
            <p
                class="-mt-0.5 mb-0 truncate text-2xs font-medium"
                x-text="entry.title || '—'"
            ></p>
            <x-dropdown.dropdown
                class="invisible relative z-3 ms-auto opacity-0 transition group-hover/card:visible group-hover/card:opacity-100 [&.lqd-is-active]:visible [&.lqd-is-active]:opacity-100"
                class:dropdown-dropdown="p-2"
                anchor="end"
            >
                <x-slot:trigger>
                    <x-button
                        class="inline-grid size-7 place-items-center"
                        size="none"
                        variant="ghost"
                        hover-variant="primary"
                    >
                        <x-tabler-dots class="size-5" />
                    </x-button>
                </x-slot:trigger>
                <x-slot:dropdown>
                    <template x-if="entry.status === 'completed' && entry.video_url">
                        <div>
                            <x-button
                                class="w-full justify-start !rounded-md text-start hover:transform-none"
                                tag="a"
                                variant="none"
                                hover-variant="primary"
                                ::href="entry.video_url"
                                ::download="downloadFilename(entry)"
                                @click="toggle(false)"
                            >
                                <x-tabler-download class="size-4" />
                                {{ __('Download') }}
                            </x-button>

                            <hr class="my-1">
                        </div>
                    </template>
                    <x-button
                        class="w-full justify-start !rounded-md text-start hover:transform-none"
                        variant="none"
                        hover-variant="primary"
                        @click.prevent="openRenameModal(entry); toggle(false);"
                    >
                        <x-tabler-pencil class="size-4" />
                        {{ __('Rename') }}
                    </x-button>
                    <x-button
                        class="w-full justify-start !rounded-md text-start hover:transform-none"
                        variant="none"
                        hover-variant="danger"
                        @click.prevent="deleteEntry(entry); toggle(false);"
                    >
                        <x-tabler-trash class="size-4 text-red-500 group-hover:text-current" />
                        {{ __('Delete') }}
                    </x-button>
                </x-slot:dropdown>
            </x-dropdown.dropdown>
        </div>
        <p
            class="-mt-0.5 mb-0 text-2xs opacity-50"
            x-show="entry.created_at"
            x-text="entry.created_at"
        ></p>
    </div>

    <template x-if="entry.status === 'completed' && entry.video_url">
        <button
            class="absolute inset-0 z-1"
            type="button"
            :aria-label="entry.title || '{{ __('Open video') }}'"
            @click="openVideoModal(entry)"
        ></button>
    </template>
</x-card>
