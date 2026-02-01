@extends('panel.layout.app')
@section('title', __('AI Product Photography'))
@section('titlebar_subtitle', __('State-of-the-art AI image processing for the creation and enhancement of visual content.'))

@section('content')
    <div class="py-10">
        <div class="flex flex-wrap justify-between gap-y-8">
            <form
                    class="w-full lg:w-4/12"
                    id="photo-studio-form"
                    method="post"
                    action="{{ route('dashboard.user.ai-product-shot.store') }}"
                    enctype="multipart/form-data"
            >
                @csrf
                <div
                        class="{{ old('action') == 'text_to_image' ? 'hidden' : '' }} flex w-full items-center justify-center"
                        id="img_select"
                        ondrop="dropHandler(event, 'img2img_src');"
                        ondragover="dragOverHandler(event);"
                >
                    <label
                            class="lqd-filepicker-label min-h-36 mb-5 flex w-full cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-foreground/10 bg-background p-6 text-center text-xs font-medium transition-colors hover:bg-background/80"
                            for="img2img_src"
                    >
                        <div class="flex flex-col items-center justify-center">
                            <x-tabler-camera-plus
                                    class="size-5 mb-4"
                                    stroke-width="1.5"
                            />

                            <p class="mb-0 opacity-50">
                                {{ __('Drag and drop a source image') }}
                            </p>

                            <p class="file-name mb-2 text-2xs">
                                {{ __('or click here to browse your files.') }}
                            </p>

                            <p class="mb-0 text-3xs opacity-50">
                                {{ __('(Max file size: 5MB)') }}
                            </p>
                        </div>

                        <input
                                class="hidden"
                                id="img2img_src"
                                name="image"
                                type="file"
                                accept=".png, .jpg, .jpeg"
                                onchange="handleFileSelect('img2img_src')"
                        />
                    </label>
                </div>

                <div class="space-y-5">
                    <div class="relative" x-data="{ open: false, selectedAvatar: null }">
                        <label for="background" class="block text-sm text-gray-500 mb-2">
                            {{ __('Backgrounds') }}
                        </label>
                        <div class="cursor-pointer rounded border p-2 flex items-center justify-between" @click="open = !open"
                             :class="{ 'border-blue-500': open }">
                            <div class="flex items-center">
                                <template x-if="selectedAvatar">
                                    <div class="flex items-center">
                                        <img class="mr-2 h-10 w-auto" :src="selectedAvatar.image" alt="">
                                        <span x-text="selectedAvatar.name"></span>
                                    </div>
                                </template>
                                <template x-if="!selectedAvatar">
                                    <span>{{ __('Select a Background') }}</span>
                                </template>
                            </div>
                            <svg class="h-5 w-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                            </svg>
                        </div>
                        <div class="absolute z-10 mt-1 max-h-60 w-full overflow-y-auto border bg-white dark:bg-gray-800 dark:border-gray-600"
                             x-show="open"
                             @click.away="open = false">
                            <div class="grid grid-cols-2 gap-4 p-2">
                                @foreach ($themes as $theme)
                                    <div class="flex flex-col cursor-pointer items-center hover:bg-gray-100 dark:hover:bg-heading-foreground/5 p-2"
                                         @click="selectedAvatar = { id: '{{ $theme['label'] }}', name: '{{ $theme['label'] }}', image: '{{ $theme['thumbnail'] }}' }; open = false">
                                        <img class="h-24 w-24 object-cover rounded-full shadow-md border border-gray-300 dark:border-gray-500"
                                             src="{{ $theme['thumbnail'] }}" alt="{{ $theme['label'] }}">
                                        <div class="mt-2 text-center">
                                            <span class="text-gray-900 dark:text-gray-200">{{ $theme['label'] }}</span>
                                        </div>
                                    </div>
                                @endforeach
                            </div>
                        </div>
                        <input type="hidden" name="background" :value="selectedAvatar ? selectedAvatar.id : ''">
                    </div>


                    <x-button
                            class="mt-4 w-full btn_loading"
                            size="lg"
                            type="submit"
                            id="photo-studio-form"
                    >
                        @lang('Generate')
                    </x-button>
                </div>
            </form>

            @if ($last)
                <div class="w-full lg:w-6/12">
                    <figure
                            class="relative"
                            x-data="{ showSource: false }"
                    >
                        <img
                                class="photo-studio-image-preview hidden w-full rounded-lg shadow-md"
                                id="photo-studio-image-preview"
                                :class="{ 'hidden': !showSource }"
                                src="{{ $last?->image }}"
                        />
                        <img
                                class="photo-studio-image-original w-full rounded-lg shadow-md"
                                id="photo-studio-image-original"
                                src="{{ $last?->image }}"
                                :class="{ 'hidden': showSource }"
                        />
                        <button
                                class="size-10 absolute bottom-7 end-7 inline-flex items-center justify-center rounded-full bg-background text-heading-foreground shadow transition-colors [&.active]:bg-primary [&.active]:text-primary-foreground"
                                type="button"
                                title="{{ __('Show source image') }}"
                                @click="showSource = !showSource"
                                :class="{ active: showSource }"
                        >
                            <svg
                                    width="24"
                                    height="24"
                                    viewBox="0 0 24 24"
                                    fill="currentColor"
                                    xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                        d="M8 17C7.45 17 6.97917 16.8042 6.5875 16.4125C6.19583 16.0208 6 15.55 6 15V9C6 8.45 6.19583 7.97917 6.5875 7.5875C6.97917 7.19583 7.45 7 8 7H9L10 6H14L15 7H16C16.55 7 17.0208 7.19583 17.4125 7.5875C17.8042 7.97917 18 8.45 18 9V15C18 15.55 17.8042 16.0208 17.4125 16.4125C17.0208 16.8042 16.55 17 16 17H8ZM8 15H16V9H8V15ZM12 14C12.55 14 13.0208 13.8042 13.4125 13.4125C13.8042 13.0208 14 12.55 14 12C14 11.45 13.8042 10.9792 13.4125 10.5875C13.0208 10.1958 12.55 10 12 10C11.45 10 10.9792 10.1958 10.5875 10.5875C10.1958 10.9792 10 11.45 10 12C10 12.55 10.1958 13.0208 10.5875 13.4125C10.9792 13.8042 11.45 14 12 14ZM8.55 0.5C9.11667 0.316667 9.6875 0.1875 10.2625 0.1125C10.8375 0.0375 11.4167 0 12 0C13.5667 0 15.0458 0.279167 16.4375 0.8375C17.8292 1.39583 19.0625 2.17083 20.1375 3.1625C21.2125 4.15417 22.0917 5.32083 22.775 6.6625C23.4583 8.00417 23.8667 9.45 24 11H22C21.8833 9.8 21.5667 8.67917 21.05 7.6375C20.5333 6.59583 19.8708 5.67917 19.0625 4.8875C18.2542 4.09583 17.325 3.45 16.275 2.95C15.225 2.45 14.1 2.15 12.9 2.05L14.45 3.6L13.05 5L8.55 0.5ZM15.45 23.5C14.8833 23.6833 14.3125 23.8125 13.7375 23.8875C13.1625 23.9625 12.5833 24 12 24C10.4333 24 8.95417 23.7208 7.5625 23.1625C6.17083 22.6042 4.9375 21.8292 3.8625 20.8375C2.7875 19.8458 1.90833 18.6792 1.225 17.3375C0.541667 15.9958 0.133333 14.55 0 13H2C2.13333 14.2 2.45417 15.3208 2.9625 16.3625C3.47083 17.4042 4.12917 18.3208 4.9375 19.1125C5.74583 19.9042 6.675 20.55 7.725 21.05C8.775 21.55 9.9 21.85 11.1 21.95L9.55 20.4L10.95 19L15.45 23.5Z"
                                />
                            </svg>
                        </button>
                    </figure>
                </div>
            @endif

        </div>
        <div id="preloader" class="spinner-border text-primary" role="status"
             style="display: none;width: 3rem; height: 3rem;">
            <span class="sr-only">Loading...</span>
        </div>
        @if ($images->count())
            <div
                    class="mt-16"
                    id="generator_sidebar_table"
                    x-data="{
                    modalShow: false,
                    activeItem: null,
                    activeItemId: null,
                    setActiveItem(data) {
                        this.activeItem = data;
                        this.activeItemId = data.id;
                    },
                    prevItem() {
                        const currentEl = document.querySelector(`.image-result[data-id='${this.activeItemId}']`);
                        const prevEl = currentEl?.previousElementSibling;
                        if (!prevEl) return;
                        const data = JSON.parse(prevEl.querySelector('.lqd-image-result-view').getAttribute('data-payload') || {});
                        this.setActiveItem(data);
                    },
                    nextItem() {
                        const currentEl = document.querySelector(`.image-result[data-id='${this.activeItemId}']`);
                        const nextEl = currentEl?.nextElementSibling;
                        if (!nextEl) return;
                        const data = JSON.parse(nextEl.querySelector('.lqd-image-result-view').getAttribute('data-payload') || {});
                        this.setActiveItem(data);
                    },
                }"
                    @keyup.escape.window="modalShow = false"
            >
                <h2 class="my-5">{{ __('Result') }}</h2>
                <div class="image-results grid grid-cols-2 gap-8 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                    @foreach ($images as $item)
                        <div
                                class="image-result lqd-loading-skeleton group w-full"
                                data-id="{{ $item->id }}"
                                data-generator="{{ str()->lower($item->response) }}"
                        >
                            <figure
                                    class="lqd-image-result-fig relative mb-3 aspect-square overflow-hidden rounded-lg shadow-md transition-all group-hover:-translate-y-1 group-hover:scale-105 group-hover:shadow-lg"
                                    data-lqd-skeleton-el
                            >
                                <img
                                        class="lqd-image-result-img aspect-square h-full w-full object-cover object-center"
                                        loading="lazy"
                                        src="{{ $item->image }}"
                                >
                                <div
                                        class="lqd-image-result-actions absolute inset-0 flex w-full flex-col items-center justify-center gap-2 p-4 transition-opacity group-[&.lqd-is-loading]:invisible group-[&.lqd-is-loading]:opacity-0">
                                    <div class="opacity-0 transition-opacity group-hover:opacity-100">
                                        <x-button
                                                class="lqd-image-result-download download size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white"
                                                size="none"
                                                href="{{ $item->image }}"
                                                download="{{ $item->image }}"
                                        >
                                            <x-tabler-download class="size-5"/>
                                        </x-button>
                                        <x-button
                                                class="lqd-image-result-view gallery size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white"
                                                data-payload="{{ $item }}"
                                                @click.prevent="setActiveItem( JSON.parse($el.getAttribute('data-payload') || {}) ); modalShow = true"
                                                size="none"
                                                href="#"
                                        >
                                            <x-tabler-eye class="size-5"/>
                                        </x-button>
                                        <x-button
                                                class="lqd-image-result-delete delete size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-red-500 hover:text-white"
                                                size="none"
                                                onclick="return confirm('Are you sure?')"
                                                href="{{ route('dashboard.user.ai-product-shot.delete', $item->id) }}"
                                        >
                                            <x-tabler-x class="size-4"/>
                                        </x-button>
                                    </div>
                                </div>
                            </figure>
                        </div>
                    @endforeach
                </div>

                <div
                        class="lqd-modal-img group/modal invisible fixed start-0 top-0 z-[999] flex h-screen w-screen flex-col items-center border p-3 opacity-0 [&.is-active]:visible [&.is-active]:opacity-100"
                        id="modal_image"
                        x-data
                        :class="{ 'is-active': modalShow }"
                >
                    <div
                            class="lqd-modal-img-backdrop absolute start-0 top-0 z-0 h-screen w-screen bg-black/10 opacity-0 backdrop-blur-sm transition-opacity group-[&.is-active]/modal:opacity-100"
                            @click="modalShow = false"
                    ></div>

                    <div class="lqd-modal-img-content-wrap relative z-10 my-auto max-h-[90vh] w-full">
                        <div class="container relative h-full max-w-6xl">
                            <div
                                    class="lqd-modal-img-content relative flex h-full translate-y-2 scale-[0.985] flex-wrap justify-between overflow-y-auto rounded-xl bg-background p-5 opacity-0 shadow-2xl transition-all group-[&.is-active]/modal:translate-y-0 group-[&.is-active]/modal:scale-100 group-[&.is-active]/modal:opacity-100 xl:min-h-[570px]">
                                <a
                                        class="size-9 absolute end-2 top-3 z-10 flex items-center justify-center rounded-full border bg-background text-inherit shadow-sm transition-all hover:bg-black hover:text-white dark:hover:bg-white dark:hover:text-black"
                                        @click.prevent="modalShow = false"
                                        href="#"
                                >
                                    <x-tabler-x class="size-4"/>
                                </a>

                                <figure class="lqd-modal-fig relative aspect-square min-h-[1px] w-full rounded-lg bg-cover bg-center max-md:min-h-[350px] md:w-6/12">
                                    <img
                                            class="lqd-modal-img mx-auto h-full w-auto object-cover object-center"
                                            :src="activeItem?.image"
                                            :alt="activeItem?.image"
                                    />
                                    <a
                                            class="size-9 absolute bottom-7 end-7 inline-flex items-center justify-center rounded-full bg-background text-inherit shadow-sm transition-all hover:scale-105"
                                            href="#"
                                            :href="activeItem?.image"
                                            download
                                    >
                                        <x-tabler-download class="size-4"/>
                                    </a>
                                </figure>

                                <div class="relative flex w-full flex-col p-3 md:w-5/12">
                                    <div class="relative flex flex-col items-start pb-6">
                                        <h3 class="mb-4">
                                            {{ __('Image Details') }}
                                        </h3>

                                        <span
                                                class="mb-3 inline-flex cursor-copy items-center justify-center gap-2 rounded-md bg-secondary px-2 py-1 text-center text-[11px] font-semibold text-secondary-foreground"
                                                @click="navigator.clipboard.writeText(activeItem?.input); toastr.success('{{ __('Copied prompt') }}');"
                                        >
                                            <x-tabler-copy class="size-4"/>
                                            {{ __('Prompt') }}
                                        </span>

                                        <span
                                                class="mt-2"
                                                x-text="activeItem?.input"
                                        ></span>
                                    </div>

                                </div>
                            </div>

                            <!-- Prev/Next buttons -->
                            <a
                                    class="size-9 absolute -start-1 top-1/2 z-10 inline-flex -translate-y-1/2 items-center justify-center rounded-full bg-background text-inherit shadow-md transition-all hover:scale-110 hover:bg-[--tblr-primary] hover:text-white"
                                    href="#"
                                    @click.prevent="prevItem()"
                            >
                                <x-tabler-chevron-left class="size-5"/>
                            </a>
                            <a
                                    class="size-9 absolute -end-1 top-1/2 z-10 inline-flex -translate-y-1/2 items-center justify-center rounded-full bg-background text-inherit shadow-md transition-all hover:scale-110 hover:bg-[--tblr-primary] hover:text-white"
                                    href="#"
                                    @click.prevent="nextItem()"
                            >
                                <x-tabler-chevron-right class="size-5"/>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        @endif

    </div>

    <template id="image_result">
        <div class="image-result lqd-loading-skeleton lqd-is-loading group w-full">
            <figure
                    class="lqd-image-result-fig relative mb-3 aspect-square overflow-hidden rounded-lg shadow-md transition-all group-hover:-translate-y-1 group-hover:scale-105 group-hover:shadow-lg"
                    data-lqd-skeleton-el
            >
                <img
                        class="lqd-image-result-img aspect-square h-full w-full object-cover object-center"
                        loading="lazy"
                        src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgc3R5bGU9ImZpbGw6I2VlZWVlZTsiLz48L3N2Zz4="
                >
                <div
                        class="lqd-image-result-actions absolute inset-0 flex w-full flex-col items-center justify-center gap-2 p-4 transition-opacity group-[&.lqd-is-loading]:invisible group-[&.lqd-is-loading]:opacity-0">
                    <div class="opacity-0 transition-opacity group-hover:opacity-100">
                        <x-button
                                class="lqd-image-result-download download size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white"
                                size="none"
                                href="#"
                                download=true
                        >
                            <x-tabler-download class="size-5"/>
                        </x-button>
                        <x-button
                                class="lqd-image-result-view gallery size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white"
                                @click.prevent="setActiveItem( JSON.parse($el.getAttribute('data-payload') || {}) ); modalShow = true"
                                size="none"
                                href="#"
                        >
                            <x-tabler-eye class="size-5"/>
                        </x-button>
                        <x-button
                                class="lqd-image-result-delete delete size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-red-500 hover:text-white"
                                size="none"
                                onclick="return confirm('Are you sure?')"
                                href="#"
                        >
                            <x-tabler-x class="size-4"/>
                        </x-button>
                    </div>
                    <span
                            class="lqd-image-result-type absolute bottom-4 end-4 mb-0 rounded-full bg-background px-2 py-1 text-3xs font-semibold uppercase leading-none transition-opacity group-[&.lqd-is-loading]:invisible group-[&[data-generator=de]]:text-red-500 group-[&[data-generator=sd]]:text-blue-500 group-[&.lqd-is-loading]:opacity-0"
                    ></span>
                </div>
            </figure>
            <p
                    class="lqd-image-result-title mb-1 w-full overflow-hidden overflow-ellipsis whitespace-nowrap text-heading-foreground transition-opacity"
                    data-lqd-skeleton-el
            ></p>
        </div>
    </template>
@endsection
<style>
    #preloader {
        position: fixed;
        top: 50%;
        left: 50%;
        display: none;
        width: 3rem;
        height: 3rem;
        z-index: 9999;
    }
</style>
@push('script')
    <script>
        $('#action').change(function () {
            var action = $(this).val();
            if (action == 'text_to_image') {
                $('#img_select').addClass('hidden');
            } else {
                $('#img_select').removeClass('hidden');
            }
        });

        function dropHandler(ev, id) {
            // Prevent default behavior (Prevent file from being opened)
            ev.preventDefault();
            $('#' + id)[0].files = ev.dataTransfer.files;
            $('#' + id).prev().find(".file-name").text(ev.dataTransfer.files[0].name);
        }

        function dragOverHandler(ev) {
            // Prevent default behavior (Prevent file from being opened)
            ev.preventDefault();
        }

        function handleFileSelect(id) {
            $('#' + id).prev().find(".file-name").text($('#' + id)[0].files[0].name);
        }
    </script>

    <script>
        document.getElementById('photo-studio-form').addEventListener('submit', function (event) {
            document.getElementById('preloader').style.display = 'block';
        });
    </script>
@endpush
