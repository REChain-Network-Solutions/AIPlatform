<div
    class="lqd-modal-img group/modal invisible fixed start-0 top-0 z-[999] flex h-screen w-screen flex-col items-center border p-3 opacity-0 [&.is-active]:visible [&.is-active]:opacity-100"
    x-data="{}"
    :class="{ 'is-active': modalShow }"
    @keyup.escape.window="modalShow = false"
    @keydown.left.window="if (modalShow) prevImageModal()"
    @keydown.right.window="if (modalShow) nextImageModal()"
>
    <div
        class="lqd-modal-img-backdrop absolute start-0 top-0 z-0 h-screen w-screen bg-black/10 opacity-0 backdrop-blur-sm transition-opacity group-[&.is-active]/modal:opacity-100"
        @click="modalShow = false"
    ></div>

    <div class="lqd-modal-img-content-wrap relative z-10 my-auto max-h-[90vh]">
        <div class="container relative h-full max-w-6xl">
            <div
                class="lqd-modal-img-content relative flex h-full translate-y-2 scale-[0.985] flex-wrap justify-between overflow-y-auto rounded-xl bg-background p-5 opacity-0 shadow-2xl transition-all group-[&.is-active]/modal:translate-y-0 group-[&.is-active]/modal:scale-100 group-[&.is-active]/modal:opacity-100 xl:min-h-[570px]">
                <a
                    class="absolute end-2 top-3 z-10 flex size-9 items-center justify-center rounded-full border bg-background text-inherit shadow-sm transition-all hover:bg-black hover:text-white dark:hover:bg-white dark:hover:text-black"
                    @click.prevent="modalShow = false"
                    href="#"
                >
                    <x-tabler-x class="size-4" />
                </a>

                <figure class="lqd-modal-fig relative aspect-square min-h-[1px] w-full rounded-lg bg-cover bg-center max-md:min-h-[350px] md:w-6/12">
                    <img
                        class="lqd-modal-img mx-auto h-full w-auto object-cover object-center"
                        :src="activeModal?.image"
                        :alt="activeModal?.prompt"
                        x-ref="modalImage"
                    />
                    <a
                        class="absolute bottom-7 end-7 inline-flex size-9 items-center justify-center rounded-full bg-background text-inherit shadow-sm transition-all hover:scale-105"
                        href="#"
                        :href="activeModal?.image"
                        download
                    >
                        <x-tabler-download class="size-4" />
                    </a>
                </figure>

                <div class="relative flex w-full flex-col p-3 md:w-5/12">
                    <div class="relative flex flex-col items-start pb-6">
                        <h3 class="mb-4">
                            {{ __('Image Details') }}
                        </h3>

                        <span
                            class="mb-3 inline-flex cursor-copy items-center justify-center gap-2 rounded-md bg-secondary px-2 py-1 text-center text-[11px] font-semibold text-secondary-foreground"
                            @click="navigator.clipboard.writeText(activeModal?.prompt); toastr.success('{{ __('Copied prompt') }}');"
                        >
                            <x-tabler-copy class="size-4" />
                            {{ __('Prompt') }}
                        </span>

                        <span
                            class="mt-2"
                            x-text="activeModal?.prompt"
                        ></span>
                    </div>

                    <div class="mt-auto flex flex-wrap justify-between gap-y-3 text-[13px] font-medium">
                        <div class="flex w-full md:w-[30%]">
                            <div class="lqd-modal-img-info w-full rounded-lg bg-heading-foreground/[3%] p-2.5">
                                <p class="mb-1">@lang('Date')</p>
                                <p
                                    class="mb-0 opacity-60"
                                    x-text="new Date(activeModal?.created_at).toLocaleDateString() ?? '{{ __('None') }}'"
                                ></p>
                            </div>
                        </div>
                        <div class="flex w-full md:w-[30%]">
                            <div class="lqd-modal-img-info w-full rounded-lg bg-heading-foreground/[3%] p-2.5">
                                <p class="mb-1">@lang('Resolution')</p>
                                <p
                                    class="mb-0 opacity-60"
                                    x-text="`${activeModal?.payload?.width}x${activeModal?.payload?.height}`"
                                >
                                </p>
                            </div>
                        </div>
                        <div class="flex w-full md:w-[30%]">
                            <div class="lqd-modal-img-info w-full rounded-lg bg-heading-foreground/[3%] p-2.5">
                                <p class="mb-1">@lang('AI Model')</p>
                                <p
                                    class="mb-0 w-full overflow-hidden text-ellipsis whitespace-nowrap opacity-60"
                                    :title="activeModal?.payload?.model"
                                    x-text="activeModal?.payload?.model ?? '{{ __('None') }}'"
                                ></p>
                            </div>
                        </div>
                        <div class="flex w-full md:w-[30%]">
                            <div class="lqd-modal-img-info w-full rounded-lg bg-heading-foreground/[3%] p-2.5">
                                <p class="mb-1">@lang('Image Style')</p>
                                <p
                                    class="mb-0 opacity-60"
                                    x-text="imageStyles[activeModal?.style] ?? '{{ __('None') }}'"
                                ></p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Prev/Next buttons -->
            <a
                class="absolute -start-1 top-1/2 z-10 inline-flex size-9 -translate-y-1/2 items-center justify-center rounded-full bg-background text-inherit shadow-md transition-all hover:scale-110 hover:bg-primary hover:text-primary-foreground"
                href="#"
                @click.prevent="prevImageModal()"
            >
                <x-tabler-chevron-left class="size-5" />
            </a>
            <a
                class="absolute -end-1 top-1/2 z-10 inline-flex size-9 -translate-y-1/2 items-center justify-center rounded-full bg-background text-inherit shadow-md transition-all hover:scale-110 hover:bg-primary hover:text-primary-foreground"
                href="#"
                @click.prevent="nextImageModal()"
            >
                <x-tabler-chevron-right class="size-5" />
            </a>
        </div>
    </div>
</div>
