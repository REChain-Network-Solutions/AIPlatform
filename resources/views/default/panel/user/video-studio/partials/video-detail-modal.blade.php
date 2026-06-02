<template x-teleport="body">
    <div
        class="lqd-modal-img group/modal invisible fixed start-0 top-0 z-[999] grid h-screen w-screen place-items-center px-5 opacity-0 [&.is-active]:visible [&.is-active]:opacity-100"
        :class="{ 'is-active': modalOpen }"
        @keyup.escape.window="closeVideoModal()"
    >
        <div
            class="absolute start-0 top-0 z-0 h-screen w-screen bg-black/50 opacity-0 backdrop-blur transition-opacity group-[&.is-active]/modal:opacity-100"
            @click="closeVideoModal()"
        ></div>

        <div class="relative z-10 w-[min(1100px,100%)]">
            <a
                class="absolute -end-4 -top-4 z-10 flex size-9 items-center justify-center rounded-full bg-background text-inherit shadow-sm transition-all hover:bg-red-500 hover:text-white"
                @click.prevent="closeVideoModal()"
                href="#"
            >
                <x-tabler-x class="size-4" />
            </a>

            <div class="relative flex max-h-[calc(100vh-80px)] flex-wrap overflow-y-auto rounded-xl bg-background shadow-2xl md:flex-nowrap">
                <div class="w-full p-6 md:w-7/12 md:border-e">
                    <div class="flex size-full items-center justify-center">
                        <template x-if="modalEntry?.video_url">
                            <video
                                class="max-h-full max-w-full rounded-lg"
                                x-ref="modalVideo"
                                :src="modalEntry?.video_url"
                                controls
                                autoplay
                            ></video>
                        </template>
                    </div>
                </div>

                <div class="w-full p-6 md:w-5/12">
                    <span
                        class="inline-block rounded-full bg-foreground/5 px-2 py-0.5 text-3xs font-medium"
                        x-show="modalEntry?.source_label"
                        x-text="modalEntry?.source_label"
                    ></span>

                    <h4
                        class="mt-2 text-base font-semibold"
                        x-text="modalEntry?.title || '{{ __('Video') }}'"
                    ></h4>

                    <p
                        class="mt-2 text-2xs leading-[1.4em] text-foreground/70"
                        x-show="modalEntry?.prompt"
                        x-text="modalEntry?.prompt"
                    ></p>

                    <div class="-ms-2 mt-4 flex items-center gap-0.5">
                        <x-button
                            class="size-9 bg-transparent hover:bg-foreground/5 hover:text-foreground"
                            variant="ghost"
                            hover-variant="none"
                            size="none"
                            tag="a"
                            ::href="modalEntry?.video_url"
                            ::download="modalEntry ? downloadFilename(modalEntry) : ''"
                            title="{{ __('Download') }}"
                        >
                            <x-tabler-download class="size-4" />
                        </x-button>
                        <template x-if="modalEntry?.destroy_url">
                            <x-button
                                class="size-9 bg-transparent text-red-600 hover:bg-red-50 hover:text-red-500"
                                variant="ghost"
                                hover-variant="none"
                                size="none"
                                type="button"
                                title="{{ __('Delete') }}"
                                @click="const e = modalEntry; closeVideoModal(); deleteEntry(e);"
                            >
                                <x-tabler-trash class="size-4" />
                            </x-button>
                        </template>
                    </div>

                    @includeIf('video-editor::partials.open-with-button', [
                        'url'    => 'modalEntry?.video_url',
                        'title'  => 'modalEntry?.title',
                        'width'  => 'modalEntry?.width',
                        'height' => 'modalEntry?.height',
                    ])

                    <div class="mt-10 space-y-3.5">
                        <div
                            class="flex w-full items-center justify-between gap-1 py-1.5 text-2xs font-medium"
                            x-show="modalEntry?.created_at"
                        >
                            <p class="mb-0">{{ __('Date') }}</p>
                            <p
                                class="mb-0 opacity-50"
                                x-text="modalEntry?.created_at"
                            ></p>
                        </div>
                        <div
                            class="flex w-full items-center justify-between gap-1 py-1.5 text-2xs font-medium"
                            x-show="modalEntry?.model"
                        >
                            <p class="mb-0">{{ __('AI Model') }}</p>
                            <p
                                class="mb-0 opacity-50"
                                x-text="modalEntry?.model"
                            ></p>
                        </div>
                        <div
                            class="flex w-full items-center justify-between gap-1 py-1.5 text-2xs font-medium"
                            x-show="modalEntry?.template_name"
                        >
                            <p class="mb-0">{{ __('Template') }}</p>
                            <p
                                class="mb-0 opacity-50"
                                x-text="modalEntry?.template_name"
                            ></p>
                        </div>
                        <div
                            class="flex w-full items-center justify-between gap-1 py-1.5 text-2xs font-medium"
                            x-show="modalEntry?.project_name"
                        >
                            <p class="mb-0">{{ __('Project') }}</p>
                            <p
                                class="mb-0 opacity-50"
                                x-text="modalEntry?.project_name"
                            ></p>
                        </div>
                        <div
                            class="flex w-full items-center justify-between gap-1 py-1.5 text-2xs font-medium"
                            x-show="modalEntry?.formatted_duration"
                        >
                            <p class="mb-0">{{ __('Duration') }}</p>
                            <p
                                class="mb-0 opacity-50"
                                x-text="modalEntry?.formatted_duration"
                            ></p>
                        </div>
                        <div
                            class="flex w-full items-center justify-between gap-1 py-1.5 text-2xs font-medium"
                            x-show="modalEntry?.resolution"
                        >
                            <p class="mb-0">{{ __('Resolution') }}</p>
                            <p
                                class="mb-0 opacity-50"
                                x-text="modalEntry?.resolution"
                            ></p>
                        </div>
                        <div
                            class="flex w-full items-center justify-between gap-1 py-1.5 text-2xs font-medium"
                            x-show="modalEntry?.aspect_ratio"
                        >
                            <p class="mb-0">{{ __('Ratio') }}</p>
                            <p
                                class="mb-0 opacity-50"
                                x-text="modalEntry?.aspect_ratio"
                            ></p>
                        </div>
                        <div
                            class="flex w-full items-center justify-between gap-1 py-1.5 text-2xs font-medium"
                            x-show="modalEntry?.source_language"
                        >
                            <p class="mb-0">{{ __('From') }}</p>
                            <p
                                class="mb-0 opacity-50"
                                x-text="modalEntry?.source_language"
                            ></p>
                        </div>
                        <div
                            class="flex w-full items-center justify-between gap-1 py-1.5 text-2xs font-medium"
                            x-show="modalEntry?.target_language"
                        >
                            <p class="mb-0">{{ __('To') }}</p>
                            <p
                                class="mb-0 opacity-50"
                                x-text="modalEntry?.target_language"
                            ></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
