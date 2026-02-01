@php
    $items_per_page = 5;
    $total_items = $userOpenai->total();
@endphp

<div class="flex flex-wrap gap-y-10">
    <x-card
        class="lqd-video-generator border-0 bg-[#F2F1FD] dark:bg-surface"
        size="lg"
    >
        <form
            class="flex flex-col gap-4"
            id="openai_generator_form"
            method="post"
            onsubmit="return sendOpenaiGeneratorForm();"
            enctype="multipart/form-data"
        >
            <input
                type="hidden"
                name="openai_id"
                value="{{ $openai->id }}"
            >
            @csrf
            <h3>
                {{ __('Source Video') }}
            </h3>

            <div
                class="flex w-full flex-col gap-5"
                ondrop="dropHandler(event, 'img2img_src');"
                ondragover="dragOverHandler(event);"
            >
                <label
                    class="lqd-filepicker-label flex min-h-64 w-full cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-foreground/10 bg-background text-center transition-colors hover:bg-background/80"
                    for="img2img_src"
                >
                    <div class="flex flex-col items-center justify-center py-6">
                        <x-tabler-cloud-upload
                            class="mb-4 size-11"
                            stroke-width="1.5"
                        />

                        <p class="mb-1 text-sm font-semibold">
                            {{ __('Drop your video here or browse.') }}
                        </p>

                        <p class="file-name mb-0 text-2xs">
                            {{ __('(Only .mp4, .mov, .webm, .gif will be accepted)') }}
                        </p>
                    </div>

                    <input
                        class="hidden"
                        id="img2img_src"
                        name="video"
                        type="file"
                        onchange="handleFileSelect('img2img_src')"
                    />
                </label>
            </div>

            <div class="my-2 flex flex-col flex-wrap justify-between gap-3 md:flex-row">
                <select
                    class="form-select w-full"
                    id="model"
                    name="model"
                >
                    <option value="{{ \App\Domains\Entity\Enums\EntityEnum::VIDEO_UPSCALER->value }}">{{ \App\Domains\Entity\Enums\EntityEnum::VIDEO_UPSCALER->label() }}</option>
                    <option value="{{ \App\Domains\Entity\Enums\EntityEnum::COGVIDEOX_5B }}">{{ \App\Domains\Entity\Enums\EntityEnum::COGVIDEOX_5B->label() }}</option>
                    <option value="{{ \App\Domains\Entity\Enums\EntityEnum::ANIMATEDIFF_V2V->value }}">{{ \App\Domains\Entity\Enums\EntityEnum::ANIMATEDIFF_V2V->label() }}</option>
                    <option value="{{ \App\Domains\Entity\Enums\EntityEnum::FAST_ANIMATEDIFF_TURBO->value }}">
                        {{ \App\Domains\Entity\Enums\EntityEnum::FAST_ANIMATEDIFF_TURBO->label() }}</option>
                </select>
            </div>

            <div
                class="my-2 flex hidden flex-col flex-wrap justify-between gap-3 md:flex-row"
                id="prompt_input"
            >
                <x-forms.input
                    class:container="grow"
                    id="prompt"
                    label="{{ __('Prompt') }}"
                    tooltip="{{ __('The prompt to generate the video from...') }}"
                    type="textarea"
                    placeholder="{{ __('The prompt to generate the video from...') }}"
                    name="prompt"
                    value=""
                    size="lg"
                />
                <x-forms.input
                    class:container="grow"
                    id="negative_prompt"
                    label="{{ __('Negative prompt') }}"
                    tooltip="{{ __('Provide negative prompt...') }}"
                    type="textarea"
                    placeholder="{{ __('Provide negative prompt...') }}"
                    name="negative_prompt"
                    value=""
                    size="lg"
                />
            </div>
            <div
                class="my-2 flex hidden flex-col flex-wrap justify-between gap-3 md:flex-row"
                id="first_n_seconds_input"
            >
                <x-forms.input
                    class:container="grow"
                    id="first_n_seconds"
                    label="{{ __('First N Seconds') }}"
                    tooltip="{{ __('The first N number of seconds of video to animate. Default value: 3') }}"
                    type="number"
                    name="first_n_seconds"
                    min="1"
                    max="12"
                    value="3"
                    size="lg"
                />
            </div>
            <div
                class="my-2 flex flex-col flex-wrap justify-between gap-3 md:flex-row"
                id="scale_input"
            >
                <x-forms.input
                    class:container="grow"
                    id="scale"
                    label="{{ __('Scale') }}"
                    tooltip="{{ __('The scale factor for the target video') }}"
                    type="number"
                    name="scale"
                    min="1"
                    max="8"
                    value="1"
                    size="lg"
                />
            </div>
            <div
                class="my-2 flex hidden flex-col flex-wrap justify-between gap-3 md:flex-row"
                id="num_inference_steps_input"
            >
                <x-forms.input
                    class:container="grow"
                    id="scale"
                    label="{{ __('Inference Steps') }}"
                    tooltip="{{ __('Increasing the amount of steps tells Stable Diffusion that it should take more steps to generate your final result which can increase the amount of detail in your image. Default value: 25') }}"
                    type="number"
                    name="num_inference_steps"
                    min="1"
                    max="25"
                    value="25"
                    size="lg"
                />
            </div>
            <div
                class="my-2 flex hidden flex-col flex-wrap justify-between gap-3 md:flex-row"
                id="select_every_nth_frame_input"
            >
                <x-forms.input
                    class:container="grow"
                    id="scale"
                    label="{{ __('Select Every Nth Frame') }}"
                    tooltip="{{ __('Select every Nth frame from the video. This can be used to reduce the number of frames to process, which can reduce the time and the cost. However, it can also reduce the quality of the final video. Default value: 2') }}"
                    type="number"
                    name="select_every_nth_frame"
                    min="1"
                    max="8"
                    value="2"
                    size="lg"
                />
            </div>
            <x-button
                id="openai_generator_button"
                size="lg"
                type="submit"
            >
                {{ __('Generate') }}
                <x-tabler-arrow-right class="size-5" />
            </x-button>
        </form>
    </x-card>

    <div
        class="w-full"
        x-data="{
            modalShow: false,
            activeItem: null,
            activeItemId: null,
            showInfo: false,
            originalVideoDimensions: {
                width: 0,
                height: 0,
            },
            upscaledVideoDimensions: {
                width: 0,
                height: 0,
            },
            init() {
                this.setActiveItem = this.setActiveItem.bind(this);
            },
            setActiveItem(data) {
                this.activeItem = data;
                this.activeItemId = data.id;
        
                const currenturl = window.location.href;
                const server = currenturl.split('/')[0];
                const delete_url = `${server}/dashboard/user/openai/documents/delete/image/${data.slug}`;
                deleteVideoBtn = document.querySelector(`.lqd-modal-img-content-wrap .delete`);
                deleteVideoBtn.href = delete_url;
        
                this.$nextTick(() => {
                    this.$refs.originalVideo.load();
                    this.$refs.upscaledVideo.load();
        
                    this.$refs.originalVideo.onloadedmetadata = () => {
                        this.originalVideoDimensions.width = this.$refs.originalVideo.videoWidth;
                        this.originalVideoDimensions.height = this.$refs.originalVideo.videoHeight;
                    }
        
                    this.$refs.upscaledVideo.onloadedmetadata = () => {
                        this.upscaledVideoDimensions.width = this.$refs.upscaledVideo.videoWidth;
                        this.upscaledVideoDimensions.height = this.$refs.upscaledVideo.videoHeight;
                    }
                });
            },
            prevItem() {
                const currentEl = document.querySelector(`.video-result[data-id='${this.activeItemId}']`);
                const prevEl = currentEl?.previousElementSibling;
                if (!prevEl) return;
                const data = JSON.parse(prevEl.querySelector('a.lqd-video-result-view').getAttribute('data-payload') || {});
                this.setActiveItem(data);
            },
            nextItem() {
                const currentEl = document.querySelector(`.video-result[data-id='${this.activeItemId}']`);
                const nextEl = currentEl?.nextElementSibling;
                if (!nextEl) return;
                const data = JSON.parse(nextEl.querySelector('a.lqd-video-result-view').getAttribute('data-payload') || {});
                this.setActiveItem(data);
            },
        }"
        @keyup.escape.window="modalShow = false"
    >
        <h2 class="mb-5">{{ __('Result') }}</h2>
        <div class="video-results grid grid-cols-2 gap-8 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
            @foreach ($userOpenai->take($items_per_page) as $item)
                <div
                    class="video-result group w-full"
                    data-id="{{ $item->id }}"
                >
                    <figure
                        class="lqd-video-result-fig relative aspect-square overflow-hidden rounded-lg shadow-md transition-all group-hover:-translate-y-1 group-hover:scale-105 group-hover:shadow-lg"
                    >
                        <video
                            class="lqd-video-result-video h-full w-full object-cover object-center"
                            loading="lazy"
                        >
                            <source
                                class="lqd-video-result-video-source"
                                loading="lazy"
                                src="{{ $item->output }}"
                                type="video/mp4"
                            >
                        </video>
                        <div class="lqd-video-result-actions absolute inset-0 flex w-full flex-col items-center justify-center gap-2 p-4">
                            <div class="flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                                <x-button
                                    class="lqd-video-result-play size-9 rounded-full bg-background text-foreground hover:bg-primary hover:text-primary-foreground"
                                    data-fslightbox="video-gallery"
                                    size="none"
                                    href="{{ $item->output }}"
                                >
                                    <x-tabler-player-play-filled class="size-4" />
                                </x-button>
                                <x-button
                                    class="lqd-video-result-download size-9 rounded-full bg-background text-foreground hover:bg-primary hover:text-primary-foreground"
                                    size="none"
                                    download="{{ $item->slug }}"
                                    href="{{ $item->output }}"
                                >
                                    <x-tabler-download class="size-5" />
                                </x-button>
                                <x-button
                                    class="lqd-video-result-view size-9 rounded-full bg-background text-foreground hover:bg-primary hover:text-primary-foreground"
                                    data-payload="{{ json_encode($item) }}"
                                    @click.prevent="setActiveItem( JSON.parse($el.getAttribute('data-payload') || {}) ); modalShow = true"
                                    size="none"
                                    href="#"
                                >
                                    <x-tabler-info-circle class="size-5" />
                                </x-button>
                            </div>
                        </div>
                    </figure>
                </div>
            @endforeach
        </div>

        @if ($userOpenai->count() > 0)
            <div
                class="lqd-load-more-trigger group min-h-px w-full py-8 text-center font-medium text-heading-foreground"
                data-all-loaded="false"
            >
                <span class="lqd-load-more-trigger-loading flex items-center justify-center gap-2 text-center leading-tight group-[&[data-all-loaded=true]]:hidden">
                    {{ __('Loading more') }}
                    <span class="flex gap-1">
                        @foreach ([0, 1, 2] as $item)
                            <span
                                class="inline-block h-[3px] w-[3px] animate-bounce-load-more rounded-full bg-current"
                                style="animation-delay: {{ $loop->index / 14 }}s"
                            ></span>
                        @endforeach
                    </span>
                </span>
                <span class="lqd-load-more-trigger-all-loaded hidden items-center justify-center gap-2 text-center group-[&[data-all-loaded=true]]:flex">
                    {{ __('All videos loaded') }}
                    <x-tabler-check class="size-5" />
                </span>
            </div>
        @endif

        <div
            class="lqd-modal-img fixed start-0 top-0 z-[999] hidden h-screen w-screen flex-col items-center border p-3 [&.is-active]:flex"
            :class="{ 'is-active': modalShow }"
        >
            <div
                class="lqd-modal-img-backdrop absolute start-0 top-0 z-0 h-screen w-screen bg-black/10 backdrop-blur-sm"
                @click="modalShow = false"
            ></div>

            <div class="lqd-modal-img-content-wrap relative z-10 my-auto w-full">
                <div class="container relative max-w-6xl">
                    <div class="absolute -top-4 end-0 z-10 flex flex-col gap-2 xl:-end-8 xl:top-0">
                        <x-button
                            class="size-9 rounded-full"
                            variant="ghost-shadow"
                            size="none"
                            @click.prevent="modalShow = false; showInfo = false"
                            href="#"
                            title="{{ __('Close') }}"
                        >
                            <x-tabler-x class="size-4" />
                        </x-button>

                        <div class="relative">
                            <x-button
                                class="relative z-2 size-9 rounded-full"
                                variant="ghost-shadow"
                                size="none"
                                href="#"
                                title="{{ __('View Details') }}"
                                @click.prevent="showInfo = !showInfo"
                            >
                                <x-tabler-info-circle class="size-4" />
                            </x-button>

                            <div
                                class="absolute -end-1 -top-1 z-1 flex w-52 flex-col gap-3 rounded-lg bg-background p-3 text-[12px] font-medium shadow-md xl:end-full xl:me-2"
                                x-cloak
                                x-show="showInfo"
                                x-transition
                            >
                                <div class="lqd-modal-img-info rounded-lg bg-heading-foreground/[3%] p-2.5">
                                    <p class="mb-0.5">
                                        @lang('Date')
                                    </p>
                                    <p
                                        class="mb-0 opacity-60"
                                        x-text="activeItem?.format_date ?? '{{ __('None') }}'"
                                    ></p>
                                </div>

                                <div class="lqd-modal-img-info rounded-lg bg-heading-foreground/[3%] p-2.5">
                                    <p class="mb-0.5">
                                        @lang('Original Dimensions')
                                    </p>
                                    <p
                                        class="mb-0 opacity-60"
                                        x-text="`${originalVideoDimensions.width}x${originalVideoDimensions.height}`"
                                    ></p>
                                </div>

                                <div class="lqd-modal-img-info rounded-lg bg-heading-foreground/[3%] p-2.5">
                                    <p class="mb-0.5">
                                        @lang('Upscaled To')
                                    </p>
                                    <p
                                        class="mb-0 opacity-60"
                                        x-text="`${upscaledVideoDimensions.width}x${upscaledVideoDimensions.height}` + ` (x${activeItem?.payload.scale})`"
                                    ></p>
                                </div>

                                <div class="lqd-modal-img-info rounded-lg bg-heading-foreground/[3%] p-2.5">
                                    <p class="mb-0.5">
                                        @lang('Credit')
                                    </p>
                                    <p
                                        class="mb-0 opacity-60"
                                        x-text="activeItem?.credits ?? '{{ __('None') }}'"
                                    >
                                    </p>
                                </div>
                            </div>
                        </div>

                        <x-button
                            class="delete size-9 rounded-full"
                            size="none"
                            variant="ghost-shadow"
                            hover-variant="danger"
                            href=""
                            onclick="return confirm('Are you sure?')"
                            title="{{ __('Delete Video') }}"
                        >
                            <x-tabler-trash class="size-4" />
                        </x-button>
                    </div>

                    <div class="lqd-modal-img-content relative flex max-h-[90vh] flex-wrap justify-between overflow-y-auto rounded-xl bg-background p-5 xl:min-h-[570px]">
                        <div class="grid w-full grid-cols-1 gap-4 lg:grid-cols-2">
                            <figure class="lqd-modal-fig lqd-modal-fig-upscaled relative w-full">
                                <h3 class="mb-5 text-center">
                                    @lang('Upscaled Video')
                                </h3>
                                <video
                                    class="lqd-modal-vid lqd-modal-vid-upscaled h-auto max-h-[85vh] w-full rounded-lg"
                                    loading="lazy"
                                    controls
                                    :src="activeItem?.output"
                                    x-ref="upscaledVideo"
                                >
                                    <source
                                        :src="activeItem?.output"
                                        type="video/mp4"
                                    >
                                </video>
                            </figure>

                            <figure class="lqd-modal-fig lqd-modal-fig-original relative w-full">
                                <h3 class="mb-5 text-center">
                                    @lang('Original Video')
                                </h3>
                                <video
                                    class="lqd-modal-vid lqd-modal-vid-original h-auto max-h-[85vh] w-full rounded-lg"
                                    loading="lazy"
                                    controls
                                    :src="'\/uploads\/' + activeItem?.payload?.video"
                                    x-ref="originalVideo"
                                >
                                    <source
                                        :src="'\/uploads\/' + activeItem?.payload?.video"
                                        type="video/mp4"
                                    >
                                </video>
                            </figure>
                        </div>

                    </div>

                    <!-- Prev/Next buttons -->
                    <a
                        class="absolute -start-1.5 top-1/2 z-10 inline-flex size-9 -translate-y-1/2 items-center justify-center rounded-full bg-background text-inherit shadow-sm transition-all hover:scale-110 hover:bg-primary hover:text-primary-foreground xl:-start-8"
                        href="#"
                        @click.prevent="prevItem()"
                    >
                        <x-tabler-chevron-left class="size-5" />
                    </a>
                    <a
                        class="absolute -end-1.5 top-1/2 z-10 inline-flex size-9 -translate-y-1/2 items-center justify-center rounded-full bg-background text-inherit shadow-sm transition-all hover:scale-110 hover:bg-primary hover:text-primary-foreground xl:-end-8"
                        href="#"
                        @click.prevent="nextItem()"
                    >
                        <x-tabler-chevron-right class="size-5" />
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>

<template id="prompt-template">
    <div class="each-prompt d-flex align-items-center mt-3">
        <input
            class="input-required form-control rounded-pill multi_prompts_description border border-primary bg-[#fff] text-[#000] placeholder:text-black placeholder:text-opacity-50 focus:border-white focus:bg-white dark:!border-none dark:!bg-[--lqd-header-search-bg] dark:placeholder:text-[#a5a9b1] dark:focus:!bg-[--lqd-header-search-bg]"
            type="text"
            name="titles[]"
            placeholder="Type another title or description"
            required
        >
        <button
            class="text-heading border-none bg-transparent"
            data-toggle="remove-parent"
            type="button"
        >
            <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 32 32"
                width="24px"
                height="24px"
                fill="currentColor"
            >
                <path
                    d="M 15 4 C 14.476563 4 13.941406 4.183594 13.5625 4.5625 C 13.183594 4.941406 13 5.476563 13 6 L 13 7 L 7 7 L 7 9 L 8 9 L 8 25 C 8 26.644531 9.355469 28 11 28 L 23 28 C 24.644531 28 26 26.644531 26 25 L 26 9 L 27 9 L 27 7 L 21 7 L 21 6 C 21 5.476563 20.816406 4.941406 20.4375 4.5625 C 20.058594 4.183594 19.523438 4 19 4 Z M 15 6 L 19 6 L 19 7 L 15 7 Z M 10 9 L 24 9 L 24 25 C 24 25.554688 23.554688 26 23 26 L 11 26 C 10.445313 26 10 25.554688 10 25 Z M 12 12 L 12 23 L 14 23 L 14 12 Z M 16 12 L 16 23 L 18 23 L 18 12 Z M 20 12 L 20 23 L 22 23 L 22 12 Z"
                />
            </svg>
        </button>
    </div>
</template>

<template id="video_result">
    <div class="video-result lqd-loading-skeleton lqd-is-loading group w-full">
        <figure
            class="lqd-video-result-fig relative aspect-square overflow-hidden rounded-lg shadow-md transition-all group-hover:-translate-y-1 group-hover:scale-105 group-hover:shadow-lg"
            data-lqd-skeleton-el
        >
            <video
                class="lqd-video-result-video h-full w-full object-cover object-center"
                loading="lazy"
            >
                <source
                    class="lqd-video-result-video-source"
                    src=""
                    loading="lazy"
                    type="video/mp4"
                >
            </video>
            <div class="lqd-video-result-actions absolute inset-0 flex w-full flex-col items-center justify-center gap-2 p-4">
                <div class="flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                    <x-button
                        class="lqd-video-result-play size-9 rounded-full bg-background text-foreground hover:bg-primary hover:text-primary-foreground"
                        data-fslightbox="video-gallery"
                        size="none"
                        href="#"
                    >
                        <x-tabler-player-play-filled class="size-4" />
                    </x-button>

                    <x-button
                        class="lqd-video-result-download size-9 rounded-full bg-background text-foreground hover:bg-primary hover:text-primary-foreground"
                        size="none"
                        download="true"
                        href="#"
                    >
                        <x-tabler-download class="size-5" />
                    </x-button>
                    <x-button
                        class="lqd-video-result-view size-9 rounded-full bg-background text-foreground hover:bg-primary hover:text-primary-foreground"
                        @click.prevent="setActiveItem( JSON.parse($el.getAttribute('data-payload') || {})); modalShow = true"
                        size="none"
                        href="#"
                    >
                        <x-tabler-info-circle class="size-5" />
                    </x-button>
                </div>
            </div>
        </figure>
    </div>
</template>

<svg
    width="0"
    height="0"
>
    <defs>
        <linearGradient
            id="loader-spinner-gradient"
            x1="0.667969"
            y1="6.10667"
            x2="23.0413"
            y2="25.84"
            gradientUnits="userSpaceOnUse"
        >
            <stop stop-color="#82E2F4" />
            <stop
                offset="0.502"
                stop-color="#8A8AED"
            />
            <stop
                offset="1"
                stop-color="#6977DE"
            />
        </linearGradient>
    </defs>
</svg>

<template id="video-loading-temp">
    <x-card
        class="video-loading flex w-full text-center shadow-[0_2px_2px_hsla(0,0%,0%,0.07)]"
        class:body="flex flex-col grow p-9 justify-center"
        data-video-id=""
    >
        <svg
            class="mx-auto mb-3 size-7 animate-spin"
            width="28"
            height="28"
            viewBox="0 0 28 28"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                d="M14.0013 27.3333C6.65464 27.3333 0.667969 21.3467 0.667969 14C0.667969 11.5067 1.3613 9.08 2.66797 6.97333C3.05464 6.34667 3.8813 6.16 4.50797 6.54667C5.13464 6.93333 5.3213 7.75999 4.93464 8.38665C3.89464 10.0667 3.33464 12.0133 3.33464 14C3.33464 19.88 8.1213 24.6667 14.0013 24.6667C19.8813 24.6667 24.668 19.88 24.668 14C24.668 8.12 19.8813 3.33333 14.0013 3.33333C13.268 3.33333 12.668 2.73333 12.668 2C12.668 1.26667 13.268 0.666666 14.0013 0.666666C21.348 0.666666 27.3346 6.65333 27.3346 14C27.3346 21.3467 21.348 27.3333 14.0013 27.3333Z"
                fill="url(#loader-spinner-gradient)"
            />
        </svg>
        <span class="inline-block bg-gradient-to-r from-[#82E2F4] to-[#6977DE] bg-clip-text text-sm font-semibold text-transparent">
            @lang('Processing...')
            <br>
            @lang('This may take a few mins')
        </span>
    </x-card>
</template>

@push('script')
    <script>
        $(document).ready(function() {
            $('#model').on('change', function() {
                // Get the selected value
                let selectedValue = $(this).val();

                const ANIMATEDIFF_V2V = '{{ \App\Domains\Entity\Enums\EntityEnum::ANIMATEDIFF_V2V->value }}';
                const FAST_ANIMATEDIFF_TURBO = '{{ \App\Domains\Entity\Enums\EntityEnum::FAST_ANIMATEDIFF_TURBO->value }}';
                const COGVIDEOX_5B = '{{ \App\Domains\Entity\Enums\EntityEnum::COGVIDEOX_5B->value }}';

                if (
                    selectedValue === ANIMATEDIFF_V2V ||
                    selectedValue === FAST_ANIMATEDIFF_TURBO ||
                    selectedValue === COGVIDEOX_5B
                ) {
                    $('#prompt_input').removeClass('hidden');
                    $('#scale_input').addClass('hidden');

                    if (selectedValue === FAST_ANIMATEDIFF_TURBO) {
                        $('#first_n_seconds_input').removeClass('hidden');
                    } else {
                        $('#first_n_seconds_input').addClass('hidden');
                    }


                    if (selectedValue === FAST_ANIMATEDIFF_TURBO || selectedValue === ANIMATEDIFF_V2V) {
                        $('#select_every_nth_frame_input').removeClass('hidden');
                    } else {
                        $('#select_every_nth_frame_input').addClass('hidden');
                    }

                    if (selectedValue === ANIMATEDIFF_V2V || selectedValue === COGVIDEOX_5B) {
                        $('#num_inference_steps_input').removeClass('hidden');
                    } else {
                        $('#num_inference_steps_input').addClass('hidden');
                    }

                } else {
                    $('#prompt_input').addClass('hidden');
                    $('#scale_input').removeClass('hidden');
                    $('#first_n_seconds_input').addClass('hidden');
                    $('#num_inference_steps_input').addClass('hidden');
                    $('#select_every_nth_frame_input').addClass('hidden');
                }
            });
        });


        let formData = new FormData();
        formData.append('openai_id', '{{ $openai->id }}');

        $.ajax({
            type: "post",
            headers: {
                'X-CSRF-TOKEN': "{{ csrf_token() }}",
            },
            url: "{{ route('dashboard.user.video-to-video.checked-all') }}",
            data: formData,
            contentType: false,
            processData: false,
            success: function(res) {

            }
        });


        var resizedImage;
        var imageWidth = -1;
        var imageHeight = -1;
        var postImageWidth = -1;
        var postImageHeight = -1;
        let checking = false;
        let NewResultVideoId = '';

        const currenturl = window.location.href;
        const server = currenturl.split('/')[0];

        function dropHandler(ev, id) {
            // Prevent default behavior (Prevent file from being opened)
            ev.preventDefault();
            $('#' + id)[0].files = ev.dataTransfer.files;
            resizeImage();
            $('#' + id).prev().find(".file-name").text(ev.dataTransfer.files[0].name);
        }

        function dragOverHandler(ev) {
            ev.preventDefault();
        }

        function handleFileSelect(id) {
            $('#' + id).prev().find(".file-name").text($('#' + id)[0].files[0].name);
        }

        function resizeImage(e) {

            var file;
            file = $("#img2img_src")[0].files[0];


            if (file == undefined) return;
            var reader = new FileReader();

            reader.onload = function(event) {
                var img = new Image();

                img.onload = function() {
                    var canvas = document.createElement('canvas');
                    var ctx = canvas.getContext("2d");

                    imageWidth = this.width;
                    imageHeight = this.height;

                    canvas.width = this.width;
                    canvas.height = this.height;
                    var ctx = canvas.getContext("2d");
                    ctx.drawImage(img, 0, 0, this.width, this.height);

                    var dataurl = canvas.toDataURL("image/png");

                    var byteString = atob(dataurl.split(',')[1]);
                    var mimeString = dataurl.split(',')[0].split(':')[1].split(';')[0];
                    var ab = new ArrayBuffer(byteString.length);
                    var ia = new Uint8Array(ab);
                    for (var i = 0; i < byteString.length; i++) {
                        ia[i] = byteString.charCodeAt(i);
                    }
                    var blob = new Blob([ab], {
                        type: mimeString
                    });

                    resizedImage = new File([blob], file.name);
                }
                img.src = event.target.result;
            }

            reader.readAsDataURL(file);

        }
        document.getElementById("img2img_src").addEventListener('change', resizeImage);

        (() => {
            "use strict";

            const itemsPerPage = {{ $items_per_page }};
            let offset = itemsPerPage; // Declare offset globally
            let totalItems = {{ $total_items }};
            let nextCount = Math.min(totalItems - itemsPerPage, itemsPerPage);
            let loadingQueue = [];

            const imageContainer = document.querySelector('.video-results');
            const imageResultTemplate = document.querySelector('#video_result');
            const loadMoreTrigger = document.querySelector('.lqd-load-more-trigger');
            const loadMoreObserver = new IntersectionObserver(([entry], observer) => {
                if (entry.isIntersecting) {
                    if (loadMoreTrigger.classList.contains('lqd-is-loading')) return;
                    createSkeleton(imageResultTemplate, nextCount);
                    lazyLoadImages();
                }
            }, {
                threshold: [1]
            });

            if (loadMoreTrigger) { // Check if the element exists
                loadMoreObserver.observe(loadMoreTrigger);
            }

            function createSkeleton(template, limit = 5) {
                const skeletonTemplates = [];
                for (let i = 0; i < limit; i++) {
                    const skeleton = template.content.cloneNode(true);
                    skeletonTemplates.push(skeleton);
                    imageContainer.append(skeleton);
                    loadingQueue.push(imageContainer.lastElementChild);
                }

                return skeletonTemplates;
            }

            function lazyLoadImages() {
                loadMoreTrigger.classList.add('lqd-is-loading');

                fetch(`{{ route('dashboard.user.openai.lazyloadimage') }}?offset=${offset}&post_type=ai_video_to_video`)
                    .then(response => response.json())
                    .then(data => {
                        const videos = data.images;
                        const currenturl = window.location.href;
                        const server = currenturl.split('/')[0];

                        nextCount = Math.min(data.count_remaining, itemsPerPage);

                        videos.forEach((video, index) => {
                            const videoResultTemplate = loadingQueue[index];
                            const delete_url = `${server}/dashboard/user/openai/documents/delete/image/${video.slug}`;

                            videoResultTemplate.setAttribute('data-id', video.id);
                            videoResultTemplate.querySelector('.lqd-video-result-video').setAttribute('src', video.output);
                            videoResultTemplate.querySelector('.lqd-video-result-video-source').setAttribute('src', video.output);
                            videoResultTemplate.querySelector('.lqd-video-result-view').setAttribute('data-payload', JSON.stringify(video));

                            videoResultTemplate.querySelector('.lqd-video-result-download').setAttribute('href', video.output);
                            videoResultTemplate.querySelector('.lqd-video-result-download').setAttribute('download', video.slug);
                            videoResultTemplate.querySelector('.lqd-video-result-play').setAttribute('href', video.output);

                            videoResultTemplate.classList.remove('lqd-is-loading');
                        });

                        loadingQueue = [];

                        // Update the offset for the next lazy loading request
                        offset += videos.length;
                        // Refresh lightbox, check if there are more videos
                        refreshFsLightbox();

                        loadMoreTrigger.classList.remove('lqd-is-loading');

                        if (data.count_remaining <= 0) {
                            loadMoreTrigger.setAttribute('data-all-loaded', 'true');
                            loadMoreObserver.disconnect();
                        } else {
                            // check if loadMoreTrigger is in view. if it is load more images
                            if (loadMoreTrigger.getBoundingClientRect().top <= window.innerHeight) {
                                createSkeleton(imageResultTemplate, nextCount);
                                lazyLoadImages();
                            }
                        }

                    });
            }
        })();

        function checkVideoDone() {
            'use strict';
            if (checking) return;
            checking = true;

            let formData = new FormData();
            formData.append('id', NewResultVideoId);
            formData.append('url', sourceImgUrl);
            formData.append('size', `${postImageWidth}x${postImageHeight}`);

            $.ajax({
                type: "post",
                headers: {
                    'X-CSRF-TOKEN': "{{ csrf_token() }}",
                },
                url: "{{ route('dashboard.user.video-to-video.checked') }}",
                data: formData,
                contentType: false,
                processData: false,
                success: function(res) {
                    checking = false;
                    if (res.status === 'finished') {
                        toastr.success('Video generated successfully');

                        clearInterval(intervalId);
                        intervalId = -1;
                        const videoContainer = document.querySelector('.video-results');
                        const videoResultTemplate = document.querySelector('#video_result').content.cloneNode(true);
                        const delete_url = `${server}/dashboard/user/openai/documents/delete/image/${res.video.slug}`;

                        videoResultTemplate.querySelector('.video-result').classList.remove('lqd-is-loading');
                        videoResultTemplate.querySelector('.video-result').setAttribute('data-id', res.video.id);
                        videoResultTemplate.querySelector('.lqd-video-result-video source').setAttribute('src', res.video.output);
                        videoResultTemplate.querySelector('.lqd-video-result-view').setAttribute('data-payload', JSON.stringify(res.video));

                        videoResultTemplate.querySelector('.lqd-video-result-download').setAttribute('href', res.video.output);
                        videoResultTemplate.querySelector('.lqd-video-result-download').setAttribute('download', res.video.slug);
                        videoResultTemplate.querySelector('.lqd-video-result-play').setAttribute('href', res.video.output);

                        document.querySelector(`.video-loading[data-video-id='${res.video.id}']`)?.remove();

                        videoContainer.insertBefore(videoResultTemplate, videoContainer.firstChild);

                        $('#openai_generator_button').html('{{ __('Generate') }}');
                        $('#openai_generator_button').prop('disabled', false);
                        // hideLoadingIndicators();

                        refreshFsLightbox();
                    } else if (res.status == 'in-progress') {

                    }
                },
                error: function(data) {
                    checking = false;
                    clearInterval(intervalId);
                    document.getElementById("openai_generator_button").disabled = false;
                    document.getElementById("openai_generator_button").innerHTML = '{{ __('Generate') }}';
                    Alpine.store('appLoadingIndicator').hide();
                    document.querySelector('#workbook_regenerate')?.classList?.add('hidden');

                    document.querySelector(`.video-loading[data-video-id='${NewResultVideoId}']`)?.remove();

                    if (data.responseJSON.errors) {
                        $.each(data.responseJSON.errors, function(index, value) {
                            toastr.error(value);
                        });
                    } else if (data.responseJSON.message) {
                        toastr.error(data.responseJSON.message);
                    }
                }
            });
        }


        function sendOpenaiGeneratorForm(ev) {
            ev?.preventDefault();
            ev?.stopPropagation();

            document.getElementById("openai_generator_button").disabled = true;
            document.getElementById("openai_generator_button").innerHTML = magicai_localize.please_wait;

            Alpine.store('appLoadingIndicator').show();

            var formData = new FormData();
            formData.append('post_type', '{{ $openai->slug }}');
            formData.append('openai_id', {{ $openai->id }});
            formData.append('custom_template', {{ $openai->custom_template }});
            formData.append('model', $('#model').val());
            formData.append('scale', $('#scale').val());
            formData.append('prompt', $('#prompt').val());
            formData.append('first_n_seconds', $('#first_n_seconds').val());
            formData.append('negative_prompt', $('#negative_prompt').val());
            formData.append('num_inference_steps', $('#num_inference_steps').val());
            formData.append('select_every_nth_frame', $('#select_every_nth_frame').val());
            var fileData = $('#img2img_src')[0].files[0];
            const loadingEl = document.querySelector('#video-loading-temp').content.cloneNode(true).firstElementChild;
            const videoContainer = document.querySelector('.video-results');

            const loadingTemplateEl = videoContainer.insertAdjacentElement('afterbegin', loadingEl);

            formData.append("video", fileData);

            $.ajax({
                type: "post",
                headers: {
                    'X-CSRF-TOKEN': "{{ csrf_token() }}",
                    'Accept': 'application/json',
                },
                url: "{{ route('dashboard.user.video-to-video.generate') }}",
                data: formData,
                contentType: false,
                processData: false,
                success: function(res) {

                    if (res.status !== 'success' && (res.message)) {
                        toastr.error(res.message);
                        loadingEl.remove();
                        hideLoadingIndicators();
                        return;
                    }

                    NewResultVideoId = res.id;
                    loadingEl.setAttribute('data-video-id', res.id);

                    console.log(res);

                    setTimeout(function() {
                        sourceImgUrl = res.sourceUrl;
                        intervalId = setInterval(checkVideoDone, 10000);
                    }, 750);
                },
                error: function(data) {
                    console.log(data);
                    document.getElementById("openai_generator_button").disabled = false;
                    document.getElementById("openai_generator_button").innerHTML = '{{ __('Genarate') }}';
                    Alpine.store('appLoadingIndicator').hide();
                    document.querySelector('#workbook_regenerate')?.classList?.add('hidden');
                    if (data.responseJSON.errors) {
                        $.each(data.responseJSON.errors, function(index, value) {
                            toastr.error(value);
                        });
                    } else if (data.responseJSON.message) {
                        toastr.error(data.responseJSON.message);
                    }
                }
            });
            return false;
        }
    </script>
@endpush
