@php use Illuminate\Support\Str; @endphp
<div
    class="fixed inset-0 z-[999] flex items-center justify-center bg-black/50"
    id="mediaManagerModal"
    wire:ignore.self
    x-data="{ show: @entangle('showModal') }"
    x-show="show"
    x-transition.opacity
    x-on:click.self="$wire.closeModal()"
    x-cloak
>
    <div class="container relative">
        <x-card
            class="relative flex max-h-[90vh] min-h-96 flex-col overscroll-contain rounded-xl p-5 shadow-xl"
            class:body="static flex flex-col overflow-hidden rounded-lg border p-0"
            size="lg"
        >
            <button
                class="absolute -end-3 -top-3 inline-grid size-8 place-items-center rounded-full bg-card-background shadow-lg shadow-black/5 transition md:-end-4 md:-top-4"
                type="button"
                wire:click="closeModal"
                aria-label="{{ __('Close') }}"
            >
                <x-tabler-x class="size-5" />
            </button>

            {{-- Header with Title, Filters, Sorting, and Search --}}
            <x-slot:head
                class="mb-5 w-full rounded-lg border"
            >
                <h2 class="mb-4 text-xl font-semibold">@lang('Content Manager')</h2>

                <div class="flex w-full justify-between text-xs max-md:flex-col max-md:gap-2">
                    <p class="-mb-3 hidden text-xs opacity-80 max-md:block">
                        @lang('Filters'):
                    </p>
                    <div
                        class="relative md:-ms-3"
                        x-data="{ open: false }"
                        @click.outside="open = false"
                    >
                        <button
                            class="hidden w-full items-center gap-1.5 py-2 text-sm font-medium max-md:flex"
                            @click.prevent="open = !open"
                            type="button"
                        >
                            {{ $activeFilter }}
                            <x-tabler-chevron-down
                                class="size-4 transition"
                                ::class="{ 'rotate-180': open }"
                            />
                        </button>
                        {{-- Filters Tabs --}}
                        <ul
                            class="w-full grow text-sm transition-all max-md:pointer-events-none max-md:invisible max-md:absolute max-md:start-0 max-md:top-full max-md:z-[90] max-md:mt-1 max-md:min-w-[min(195px,100%)] max-md:translate-y-1 max-md:rounded-dropdown max-md:border max-md:border-dropdown-border max-md:bg-dropdown-background max-md:p-2 max-md:opacity-0 max-md:shadow-lg max-md:shadow-black/5 max-md:before:absolute max-md:before:inset-x-0 max-md:before:-top-1 max-md:before:bottom-full md:flex md:flex-wrap md:items-center md:justify-start md:gap-2 xl:gap-5 max-md:[&.open]:pointer-events-auto max-md:[&.open]:visible max-md:[&.open]:translate-y-0 max-md:[&.open]:opacity-100"
                            :class="{ 'open': open }"
                        >
                            @foreach ($filters as $filter)
                                <li>
                                    <button
                                        wire:click="changeFilter('{{ $filter }}'); open = false;"
                                        @class([
                                            'rounded-full px-3 py-2 leading-tight transition-all hover:bg-accent/80 hover:text-accent-foreground max-md:w-full max-md:rounded-md max-md:text-start [&.lqd-is-active]:shadow-md [&.lqd-is-active]:shadow-black/5 [&.lqd-is-active]:bg-accent [&.lqd-is-active]:text-accent-foreground',
                                            'lqd-is-active' => $activeFilter === $filter,
                                        ])
                                    >
                                        @lang($filter)
                                    </button>
                                </li>
                            @endforeach
                        </ul>
                    </div>
                    {{-- Sorting & Search --}}
                    <div class="flex items-center gap-2">
                        <div class="flex text-nowrap text-sm font-medium text-heading-foreground">
                            <x-dropdown.dropdown
                                class:dropdown-dropdown="max-lg:end-auto max-lg:start-0"
                                :teleport="false"
                                offsetY="1rem"
                                anchor="end"
                            >
                                <x-slot:trigger
                                    class="whitespace-nowrap px-0 py-1"
                                    variant="link"
                                    size="xs"
                                >
                                    {{ __('Sort by') }}
                                    <x-tabler-chevron-down
                                        class="size-4 transition"
                                        ::class="{ 'rotate-180': open }"
                                    />
                                </x-slot:trigger>

                                <x-slot:dropdown
                                    class="overflow-hidden text-2xs font-medium"
                                >
                                    <div class="lqd-sort-list flex flex-col">
                                        @foreach ($sortButtons as $button)
                                            <button
                                                class="{{ $sort === $button['sort'] ? 'bg-foreground/5' : '' }} group flex w-full items-center gap-1 px-3 py-2 hover:bg-foreground/5"
                                                wire:click.prevent="changeSort('{{ $button['sort'] }}')"
                                            >
                                                {{ $button['label'] }}
                                                @if ($sort === $button['sort'])
                                                    <x-tabler-caret-down-filled class="{{ $sortAscDesc === 'asc' ? 'rotate-180' : '' }} size-3 opacity-80" />
                                                @endif
                                            </button>
                                        @endforeach
                                    </div>
                                </x-slot:dropdown>
                            </x-dropdown.dropdown>
                        </div>
                        {{-- Search input with clear button --}}
                        <div class="relative">
                            <x-tabler-search
                                class="pointer-events-none absolute start-3 top-1/2 z-10 size-4 -translate-y-1/2 opacity-75"
                                stroke-width="1.5"
                            />
                            <x-forms.input
                                class="min-w-48 border-none bg-heading-foreground/5 ps-10 transition-colors max-lg:rounded-md"
                                id="serach-resources"
                                container-class="peer"
                                wire:model.live.debounce.300ms="searchTerm"
                                type="text"
                                placeholder="{{ __('Search') }}"
                            />
                            @if (!empty($searchTerm))
                                <button
                                    class="absolute right-2 top-1/2 -translate-y-1/2 transform text-gray-400 hover:text-gray-600"
                                    wire:click="clearSearch"
                                    type="button"
                                >
                                    <x-tabler-x class="size-3" />
                                </button>
                            @endif
                            {{-- Search loading indicator --}}
                            <div
                                class="absolute right-2 top-1/2 -translate-y-1/2 transform"
                                wire:loading
                                wire:target="searchTerm"
                            >
                                <div class="h-3 w-3 animate-spin rounded-full border-b border-blue-500"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </x-slot:head>

            {{-- Upload Files Section --}}
            @if ($activeFilter === 'Upload Files')
                <div class="p-4">
                    {{-- Upload Area --}}
                    <div
                        class="group/drop-area relative w-full rounded-3xl border-2 border-dashed px-5 py-8 transition-colors"
                        x-data="mediaManagerUploadArea"
                        @dragover="handleDragOver($event)"
                        @dragleave="handleDragLeave($event)"
                        @drop="handleDrop($event)"
                        :class="{ 'border-blue-500 bg-blue-50': dragover }"
                    >
                        <div class="mx-auto space-y-4 text-center">
                            <div class="mx-auto mb-4 inline-grid w-12 place-content-center">
                                <x-tabler-circle-arrow-up
                                    class="size-12 text-heading-foreground opacity-60"
                                    stroke-width="1.5"
                                />
                            </div>

                            <h4 class="text-base">@lang('Drag and Drop a File')</h4>

                            <div class="mx-auto flex w-[300px] items-center gap-7 text-2xs font-medium text-heading-foreground">
                                <span class="inline-flex h-px grow bg-heading-foreground/5"></span>
                                @lang('or')
                                <span class="inline-flex h-px grow bg-heading-foreground/5"></span>
                            </div>

                            {{-- File Processing Indicator --}}
                            <div
                                class="w-full px-8"
                                x-show="fileProcessing && (!$wire.uploadingFiles || $wire.uploadingFiles.length === 0)"
                                x-transition.opacity
                            >
                                <div class="rounded-lg border border-blue-200 bg-blue-50 p-3">
                                    <div class="flex items-center justify-center gap-2">
                                        <div class="h-4 w-4 animate-spin rounded-full border-b-2 border-blue-500"></div>
                                        <span class="text-xs text-blue-600">@lang('Processing selected files')...</span>
                                    </div>
                                    <p class="mt-1 text-center text-xs text-blue-500">@lang('Please wait, validating large files may take a moment.')</p>
                                </div>
                            </div>

                            {{-- File List --}}
                            @if (count($uploadingFiles))
                                <div
                                    class="w-full px-8"
                                    x-data
                                    x-init="$el.closest('[x-data]').fileProcessing = false"
                                >
                                    <ul class="mt-2 space-y-1 rounded-lg bg-gray-50 p-3 text-left text-xs text-gray-600">
                                        @foreach ($uploadingFiles as $file)
                                            <li class="flex items-center justify-between">
                                                <span>{{ $file?->getClientOriginalName() }}</span>
                                                <span class="text-gray-400">{{ number_format($file?->getSize() / 1024, 1) }} KB</span>
                                            </li>
                                        @endforeach
                                    </ul>
                                </div>
                            @endif

                            {{-- Upload Errors --}}
                            @if (!empty($uploadErrors))
                                <div class="w-full px-8">
                                    <div class="rounded-lg border border-red-200 bg-red-50 p-3">
                                        <ul class="space-y-1 text-xs text-red-600">
                                            @foreach ($uploadErrors as $error)
                                                <li>• {{ $error }}</li>
                                            @endforeach
                                        </ul>
                                    </div>
                                </div>
                            @endif

                            {{-- Upload Progress --}}
                            @if ($isUploading && !empty($uploadProgress))
                                <div class="w-full px-8">
                                    <div class="rounded-lg border border-blue-200 bg-blue-50 p-3">
                                        <div class="flex items-center gap-2">
                                            <div class="h-4 w-4 animate-spin rounded-full border-b-2 border-blue-500"></div>
                                            <span class="text-xs text-blue-600">{{ $uploadProgress }}</span>
                                        </div>
                                    </div>
                                </div>
                            @endif

                            {{-- Upload Success --}}
                            @if (!empty($uploadedFiles) && !$isUploading)
                                <div class="w-full px-8">
                                    <div class="rounded-lg border border-green-200 bg-green-50 p-3">
                                        <p class="text-xs font-medium text-green-600">✓ {{ count($uploadedFiles) }} file(s) uploaded successfully!</p>
                                        <ul class="mt-1 space-y-1 text-xs text-green-600">
                                            @foreach ($uploadedFiles as $file)
                                                <li>• {{ $file['name'] }}</li>
                                            @endforeach
                                        </ul>
                                    </div>
                                </div>
                            @endif

                            {{-- Action Buttons --}}
                            <div>
                                <input
                                    class="hidden"
                                    id="fileInput"
                                    data-exclude-media-manager="true"
                                    type="file"
                                    multiple
                                    wire:model="uploadingFiles"
                                    x-ref="fileInput"
                                    @change="handleFileSelect()"
                                    accept="*"
                                />

								<x-button
									type="button"
									onclick="document.getElementById('fileInput').click()"
									variant="outline"
									:disabled="$isUploading"
								>
									{{ count($uploadingFiles) ? __('Add More') : __('Browse Files') }}
								</x-button>
                            </div>

                            <p class="m-0 text-3xs font-medium opacity-60">
                                @lang('Max :max files, :maxSize MB each.', [
                                    'max' => setting('media_max_files', 5),
                                    'maxSize' => setting('media_max_size', 25),
                                ])
                                <br>
                                @lang('Allowed types: :types', [
                                    'types' => setting('media_allowed_types', 'jpg, png, gif, webp, svg, mp4, avi, mov, wmv, flv, webm, mp3, wav, m4a, pdf, doc, docx, xls, xlsx'),
                                ])
                            </p>
                        </div>
                    </div>
                </div>
            @endif

            {{-- Images Section --}}
            @if ($activeFilter === 'Images')
                <div
                    class="overflow-y-auto overscroll-contain p-4"
                    x-data="infiniteScroll"
                    x-ref="scrollContainer"
                >
                    {{-- Search results info --}}
                    @if (!empty($searchTerm))
                        <div class="mb-4 text-sm text-gray-600">
                            @if ($images->count() > 0)
                                @lang('Found :count images for', ['count' => $images->count()]) "<strong>{{ $searchTerm }}</strong>"
                            @else
                                @lang('No images found for') "<strong>{{ $searchTerm }}</strong>"
                            @endif
                        </div>
                    @endif

                    <div
                        class="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6"
                        wire:loading.class="opacity-50"
                    >
                        @forelse ($images as $image)
                            @php
                                $isDisabledImage = $this->isCardDisabled('image', $this->allowedTypes);
                            @endphp
                            <div
                                class="{{ in_array((string) $image->id, $selectedImages ?? [], true) ? 'outline outline-[3px] outline-accent' : '' }} {{ $isDisabledImage ? 'pointer-events-none opacity-30 bg-gray-50 border-gray-200 cursor-not-allowed filter grayscale' : 'cursor-pointer hover:bg-gray-100 hover:shadow-md hover:border-gray-300' }} group relative w-fit overflow-hidden rounded-lg border p-2.5 transition-all duration-200"
                                @if (!$isDisabledImage && $allowMultipleSelection) wire:click="toggleSelect('image', '{{ $image->id }}')"
								@elseif (!$isDisabledImage)
									wire:click="$set('selectedImages', ['{{ $image->id }}'])" @endif
                                tabindex="{{ $isDisabledImage ? '-1' : '0' }}"
                                role="button"
                                @if ($isDisabledImage) aria-disabled="true" @endif
                            >
                                <div class="relative overflow-hidden rounded-lg">

									@php
										$img = $image->output_url ?? $image->url;
										if (Str::contains($img, 'uploads')) {
											$img = Str::after($img, '/uploads');
											$img = '/uploads' . $img;
										}
									@endphp
                                    <img
                                        class="h-32 w-32 object-cover"
                                        src="{{ ThumbImage($img) }}"
                                        alt="{{ $image->title ?? $image->input }}"
                                        loading="lazy"
                                        onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik0xMiAxNkM4LjY4NjI5IDE2IDYgMTMuMzEzNyA2IDEwQzYgNi42ODYyOSA4LjY4NjI5IDQgMTIgNEMxNS4zMTM3IDQgMTggNi42ODYyOSAxOCAxMEMxOCAxMy4zMTM3IDE1LjMxMzcgMTYgMTIgMTZaIiBmaWxsPSIjOUI5QkEwIi8+CjxwYXRoIGQ9Ik0xMiAxMkMxMC44OTU0IDEyIDEwIDExLjEwNDYgMTAgMTBDMTAgOC44OTU0MyAxMC44OTU0IDggMTIgOEMxMy4xMDQ2IDggMTQgOC44OTU0MyAxNCAxMEMxNCAxMS4xMDQ2IDEzLjEwNDYgMTIgMTIgMTJaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K'"
                                    />
                                    <div
                                        class="{{ in_array((string) $image->id, $selectedImages ?? [], true) ? 'opacity-100' : '' }} pointer-events-none absolute bottom-0 left-0 right-0 truncate bg-background p-1 text-xs text-heading-foreground opacity-0 transition-opacity duration-300 group-hover:opacity-100"
                                        style="pointer-events:none;"
                                    >
                                        {{ $image->title ?? Str::limit($image->input ?? ($image->filename ?? 'Untitled'), 30) }}
                                    </div>

                                    {{-- Show creation date on hover --}}
                                    <div
                                        class="absolute right-1 top-1 rounded bg-background px-1 py-0.5 text-xs text-heading-foreground opacity-0 transition-opacity group-hover:opacity-100">
                                        {{ $image->format_date ?? $image->created_at->format('M d, Y') }}
                                    </div>

                                    {{-- Selection indicator --}}
                                    @if (in_array((string) $image->id, $selectedImages ?? [], true))
                                        <span class="absolute end-1 top-1.5 flex items-center justify-center rounded-full bg-background p-2 shadow-lg">
                                            <x-tabler-check class="size-4" />
                                        </span>
                                    @endif
                                </div>
                            </div>
                        @empty
                            <div class="col-span-6 py-8 text-center text-gray-500">
                                <x-tabler-photo class="mx-auto mb-2 size-12 opacity-50" />
                                <p>@lang('No images found.')</p>
                                @if (!empty($searchTerm))
                                    <p class="mt-1 text-sm">@lang('Try adjusting your search term.')</p>
                                    <button
                                        class="mt-2 text-sm text-blue-500 underline hover:text-blue-700"
                                        wire:click="clearSearch"
                                    >
                                        @lang('Clear search')
                                    </button>
                                @endif
                            </div>
                        @endforelse
                    </div>

                    {{-- Loading indicator --}}
                    @if ($hasMoreImages && !$isLoading && $images->count() > 0)
                        <div
                            class="flex justify-center py-4"
                            x-ref="loadTrigger"
                        >
                            <div class="text-sm text-gray-500">
                                @lang('Scroll for more images...')
                            </div>
                        </div>
                    @endif

                    {{-- Loading spinner --}}
                    <div
                        class="flex items-center justify-center py-4"
                        wire:loading
                    >
                        <x-tabler-loader-2 class="size-6 animate-spin"></x-tabler-loader-2>
                    </div>
                </div>

                @if (!empty($selectedImages) && count($selectedImages) > 0)
                    <div class="flex justify-end border-t border-gray-200 px-4 pb-4 pt-4">
                        <x-button
                            class="w-full p-2 opacity-90 hover:opacity-100"
                            wire:click="insertSelected('image')"
                            type="button"
                            size="sm"
                            variant="secondary"
                        >
                            @lang($allowMultipleSelection ? 'Insert Selected Images' : 'Insert Selected Image')
                            @if ($allowMultipleSelection)
                                ({{ count($selectedImages) }})
                            @endif
                        </x-button>
                    </div>
                @endif
            @endif

            {{-- Videos Section --}}
            @if ($activeFilter === 'Videos')
                <div
                    class="overflow-y-auto overscroll-contain p-4"
                    x-data="infiniteScroll"
                    x-ref="scrollContainer"
                >
                    {{-- Search results info --}}
                    @if (!empty($searchTerm))
                        <div class="mb-4 text-sm text-gray-600">
                            @if ($videos->count() > 0)
                                @lang('Found :count videos for', ['count' => $videos->count()]) "<strong>{{ $searchTerm }}</strong>"
                            @else
                                @lang('No videos found for') "<strong>{{ $searchTerm }}</strong>"
                            @endif
                        </div>
                    @endif

                    <div
                        class="grid grid-cols-2 items-center gap-4 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6"
                        wire:loading.class="opacity-50"
                    >
                        @forelse ($videos as $video)
                            @php
                                $isDisabledVideo = $this->isCardDisabled('video', $this->allowedTypes);
                            @endphp
                            <div
                                class="{{ in_array((string) $video->id, $selectedVideos ?? [], true) ? 'outline outline-[3px] outline-accent' : '' }} {{ $isDisabledVideo ? 'pointer-events-none opacity-30 bg-gray-50 border-gray-200 cursor-not-allowed filter grayscale' : 'cursor-pointer hover:bg-gray-100 hover:shadow-md hover:border-gray-300' }} group relative w-fit items-center justify-center overflow-hidden rounded-lg border p-2.5 transition-all duration-200"
                                @if (!$isDisabledVideo && $allowMultipleSelection) wire:click="toggleSelect('video', '{{ $video->id }}')"
								@elseif (!$isDisabledVideo)
									wire:click="$set('selectedVideos', ['{{ $video->id }}'])" @endif
                                tabindex="{{ $isDisabledVideo ? '-1' : '0' }}"
                                role="button"
                                @if ($isDisabledVideo) aria-disabled="true" @endif
                                x-data="{
                                    isPlaying: false,
                                    showControls: false,
                                    videoLoaded: false,
                                    loadVideo() {
                                        if (!this.videoLoaded) {
                                            this.$refs.videoPlayer.src = '{{ $video->output_url }}';
                                            this.videoLoaded = true;
                                        }
                                    }
                                }"
                            >
                                <div class="relative overflow-hidden rounded-lg">
                                    {{-- Video thumbnail or video player --}}
                                    @if (isset($video->output_url) && $video->output_url)
                                        <div class="relative h-32 w-32">
                                            {{-- Static thumbnail/placeholder shown initially --}}
                                            <div
                                                class="absolute inset-0 flex items-center justify-center bg-gray-200"
                                                x-show="!videoLoaded"
                                            >
                                                <x-tabler-video class="size-8 text-gray-400" />
                                            </div>

                                            {{-- Video element (initially without src) --}}
                                            <video
                                                class="h-32 w-32 object-cover"
                                                muted
                                                preload="none"
                                                x-ref="videoPlayer"
                                                x-show="videoLoaded"
                                                @mouseenter="showControls = true"
                                                @mouseleave="showControls = false"
                                                @click.stop
                                                @ended="isPlaying = false"
                                                @play="isPlaying = true"
                                                @pause="isPlaying = false"
                                                @loadeddata="$refs.videoPlayer.currentTime = 0"
                                            >
                                                {{-- Source will be added dynamically via Alpine.js --}}
                                            </video>
                                        </div>
                                    @else
                                        <div class="flex h-32 w-32 items-center justify-center bg-gray-200">
                                            <x-tabler-video class="size-8 text-gray-400" />
                                        </div>
                                    @endif

                                    {{-- Video Controls Overlay --}}
                                    @if (isset($video->output_url) && $video->output_url)
                                        <div
                                            class="absolute inset-0 flex items-center justify-center transition-opacity"
                                            x-show="showControls || !isPlaying"
                                            x-transition:enter="transition-opacity duration-200"
                                            x-transition:enter-start="opacity-0"
                                            x-transition:enter-end="opacity-100"
                                            x-transition:leave="transition-opacity duration-200"
                                            x-transition:leave-start="opacity-100"
                                            x-transition:leave-end="opacity-0"
                                        >
                                            <div class="rounded-full bg-black/50 p-2">
                                                <button
                                                    class="flex items-center justify-center text-white transition-colors hover:text-gray-300"
                                                    @click.stop="
                                            if (!videoLoaded) {
                                                loadVideo();
                                                // Wait a bit for video to load before playing
                                                setTimeout(() => {
                                                    $refs.videoPlayer.play();
                                                }, 100);
                                            } else if (isPlaying) {
                                                $refs.videoPlayer.pause();
                                            } else {
                                                $refs.videoPlayer.play();
                                            }
                                        "
                                                >
                                                    {{-- Show loading spinner when video is being loaded --}}
                                                    <div
                                                        class="h-6 w-6 animate-spin rounded-full border-2 border-white border-t-transparent"
                                                        x-show="videoLoaded && !isPlaying && $refs.videoPlayer && $refs.videoPlayer.readyState < 3"
                                                    ></div>

                                                    <x-tabler-player-play
                                                        class="size-6"
                                                        x-show="!isPlaying && (!videoLoaded || ($refs.videoPlayer && $refs.videoPlayer.readyState >= 3))"
                                                    />
                                                    <x-tabler-player-pause
                                                        class="size-6"
                                                        x-show="isPlaying"
                                                    />
                                                </button>
                                            </div>
                                        </div>
                                    @endif

                                    <div
                                        class="pointer-events-none absolute bottom-0 left-0 right-0 z-1 truncate bg-background p-1 text-xs text-heading-foreground opacity-100"
                                        style="pointer-events:none;"
                                    >
                                        {{ $video->title ?? Str::limit($video->input ?? ($video->filename ?? 'Untitled'), 30) }}
                                    </div>

                                    {{-- Show creation date on hover --}}
                                    <div
                                        class="absolute right-1 top-1 rounded bg-background px-1 py-0.5 text-xs text-heading-foreground opacity-0 transition-opacity group-hover:opacity-100">
                                        {{ $video->format_date ?? $video->created_at->format('M d, Y') }}
                                    </div>

                                    {{-- Selection indicator --}}
                                    @if (in_array((string) $video->id, $selectedVideos ?? [], true))
                                        <span class="absolute end-1 top-1.5 flex items-center justify-center rounded-full bg-background p-2 shadow-lg">
                                            <x-tabler-check class="size-4" />
                                        </span>
                                    @endif

                                    @if (!in_array((string) $video->id, $selectedVideos ?? [], true))
                                        <div class="absolute bottom-8 start-2 flex gap-1 rounded-md bg-background p-1">
                                            <x-tabler-video class="size-4" />
                                        </div>
                                    @endif
                                </div>
                            </div>
                        @empty
                            <div class="col-span-6 py-8 text-center text-gray-500">
                                <x-tabler-video class="mx-auto mb-2 size-12 opacity-50" />
                                <p>@lang('No videos found.')</p>
                                @if (!empty($searchTerm))
                                    <p class="mt-1 text-sm">@lang('Try adjusting your search term.')</p>
                                    <button
                                        class="mt-2 text-sm text-blue-500 underline hover:text-blue-700"
                                        wire:click="clearSearch"
                                    >
                                        @lang('Clear search')
                                    </button>
                                @endif
                            </div>
                        @endforelse
                    </div>

                    {{-- Loading indicator --}}
                    @if ($hasMoreVideos && !$isLoading && $videos->count() > 0)
                        <div
                            class="flex justify-center py-4"
                            x-ref="loadTrigger"
                        >
                            <div class="text-sm text-gray-500">
                                @lang('Scroll for more videos...')
                            </div>
                        </div>
                    @endif

                    {{-- Loading spinner --}}
                    <div
                        class="flex items-center justify-center py-4"
                        wire:loading
                    >
                        <x-tabler-loader-2 class="size-6 animate-spin"></x-tabler-loader-2>
                    </div>
                </div>

                @if (!empty($selectedVideos) && count($selectedVideos) > 0)
                    <div class="flex justify-end border-t border-gray-200 px-4 pb-4 pt-4">
                        <x-button
                            class="w-full p-2 opacity-90 hover:opacity-100"
                            wire:click="insertSelected('video')"
                            type="button"
                            size="sm"
                            variant="secondary"
                        >
                            @lang($allowMultipleSelection ? 'Insert Selected Videos' : 'Insert Selected Video')
                            @if ($allowMultipleSelection && count($selectedVideos))
                                ({{ count($selectedVideos) }})
                            @endif
                        </x-button>
                    </div>
                @endif
            @endif

            {{-- Other Files Section --}}
            @if ($activeFilter === 'Other Files')
                <div
                    class="overflow-y-auto overscroll-contain p-4"
                    x-data="infiniteScroll"
                    x-ref="scrollContainer"
                >
                    {{-- Search results info --}}
                    @if (!empty($searchTerm))
                        <div class="mb-4 text-sm text-gray-600">
                            @if ($otherFiles->count() > 0)
                                @lang('Found :count files for', ['count' => $otherFiles->count()]) "<strong>{{ $searchTerm }}</strong>"
                            @else
                                @lang('No files found for') "<strong>{{ $searchTerm }}</strong>"
                            @endif
                        </div>
                    @endif

                    <div
                        class="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3"
                        wire:loading.class="opacity-50"
                    >
                        @forelse ($otherFiles as $file)
                            @php
                                $isDisabledOther = $this->isCardDisabled('other', $this->allowedTypes);
                            @endphp
                            <div
                                class="{{ in_array((string) $file->id, $selectedOtherFiles ?? [], true) ? 'outline outline-[3px] outline-accent' : '' }} {{ $isDisabledOther ? 'pointer-events-none opacity-30 bg-gray-50 border-gray-200 cursor-not-allowed filter grayscale' : 'cursor-pointer hover:bg-gray-100 hover:shadow-md hover:border-gray-300' }} group relative overflow-hidden rounded-lg border p-3 transition-all duration-200"
                                @if (!$isDisabledOther && $allowMultipleSelection) wire:click="toggleSelect('other', '{{ $file->id }}')"
								@elseif (!$isDisabledOther)
									wire:click="$set('selectedOtherFiles', ['{{ $file->id }}'])" @endif
                                tabindex="{{ $isDisabledOther ? '-1' : '0' }}"
                                role="button"
                                @if ($isDisabledOther) aria-disabled="true" @endif
                            >
                                <div class="flex items-center space-x-3">
                                    {{-- File icon based on extension --}}
                                    <div class="flex-shrink-0">
                                        @if (in_array($file->extension, ['pdf']))
                                            <x-tabler-file-type-pdf class="size-8 text-red-500" />
                                        @elseif (in_array($file->extension, ['doc', 'docx']))
                                            <x-tabler-file-type-doc class="size-8 text-blue-500" />
                                        @elseif (in_array($file->extension, ['xls', 'xlsx']))
                                            <x-tabler-file-type-xls class="size-8 text-green-500" />
                                        @elseif (in_array($file->extension, ['ppt', 'pptx']))
                                            <x-tabler-file-type-ppt class="size-8 text-orange-500" />
                                        @else
                                            <x-tabler-file class="size-8 text-gray-500" />
                                        @endif
                                    </div>

                                    {{-- File info --}}
                                    <div class="min-w-0 flex-1">
                                        <p class="truncate text-sm font-medium text-gray-900">
                                            {{ \Illuminate\Support\Str::limit($file->filename, 20, '...') }}
                                        </p>
                                        <p class="truncate text-sm text-gray-500">
                                            {{ strtoupper($file->extension) }} • {{ number_format($file->file_size / 1024, 1) }} KB
                                        </p>
                                        <p class="text-xs text-gray-400">
                                            {{ $file->format_date }}
                                        </p>
                                    </div>

                                    {{-- Selection indicator --}}
                                    @if (in_array((string) $file->id, $selectedOtherFiles ?? [], true))
                                        <span class="flex items-center justify-center rounded-full bg-accent p-2 text-white">
                                            <x-tabler-check class="size-4" />
                                        </span>
                                    @endif
                                </div>
                            </div>
                        @empty
                            <div class="col-span-3 py-8 text-center text-gray-500">
                                <x-tabler-files class="mx-auto mb-2 size-12 opacity-50" />
                                <p>@lang('No files found.')</p>
                                @if (!empty($searchTerm))
                                    <p class="mt-1 text-sm">@lang('Try adjusting your search term.')</p>
                                    <button
                                        class="mt-2 text-sm text-blue-500 underline hover:text-blue-700"
                                        wire:click="clearSearch"
                                    >
                                        @lang('Clear search')
                                    </button>
                                @endif
                            </div>
                        @endforelse
                    </div>

                    {{-- Loading indicator --}}
                    @if ($hasMoreOtherFiles && !$isLoading && $otherFiles->count() > 0)
                        <div
                            class="flex justify-center py-4"
                            x-ref="loadTrigger"
                        >
                            <div class="text-sm text-gray-500">
                                @lang('Scroll for more files...')
                            </div>
                        </div>
                    @endif

                    {{-- Loading spinner --}}
                    <div
                        class="flex items-center justify-center py-4"
                        wire:loading
                    >
                        <x-tabler-loader-2 class="size-6 animate-spin"></x-tabler-loader-2>
                    </div>
                </div>

                @if (!empty($selectedOtherFiles) && count($selectedOtherFiles) > 0)
                    <div class="flex justify-end border-t border-gray-200 px-4 pb-4 pt-4">
                        <x-button
                            class="w-full p-2 opacity-90 hover:opacity-100"
                            wire:click="insertSelected('other')"
                            type="button"
                            size="sm"
                            variant="secondary"
                        >
                            @lang('Insert Selected Files') ({{ count($selectedOtherFiles) }})
                        </x-button>
                    </div>
                @endif
            @endif

            {{-- Google Drive Section --}}
            @if ($activeFilter === 'Google Drive')
                <div class="p-4">
                    <div class="py-8 text-center text-gray-500">
                        <x-tabler-brand-google-drive class="mx-auto mb-2 size-12 opacity-50" />
                        <p>@lang('Google Drive integration coming soon...')</p>
                    </div>
                </div>
            @endif

            {{-- Stock Images Section --}}
            @if ($activeFilter === 'Stock Images')
                <div class="overflow-y-auto overscroll-contain p-4">
                    {{-- Search results info --}}
                    @if (!empty($searchTerm))
                        <div class="mb-4 text-sm text-gray-600">
                            @if ($stockImages->count() > 0)
                                @lang('Found :count stock images for', ['count' => $stockImages->count()]) "<strong>{{ $searchTerm }}</strong>"
                            @else
                                @lang('No stock images found for') "<strong>{{ $searchTerm }}</strong>"
                            @endif
                        </div>
                    @endif

                    <div
                        class="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6"
                        wire:loading.class="opacity-50"
                    >
                        @forelse ($stockImages as $stockImage)
                            @php
                                $isDisabledImage = $this->isCardDisabled('image', $this->allowedTypes);
                                // Check if this stock image is selected
                                $isSelected = in_array((string) $stockImage->id, $selectedStockImages ?? [], true);
                            @endphp
                            <div
                                class="{{ $isSelected ? 'outline outline-[3px] outline-accent' : '' }} {{ $isDisabledImage ? 'pointer-events-none opacity-30 bg-gray-50 border-gray-200 cursor-not-allowed filter grayscale' : 'cursor-pointer hover:bg-gray-100 hover:shadow-md hover:border-gray-300' }} group relative w-fit overflow-hidden rounded-lg border p-2.5 transition-all duration-200"
                                @if (!$isDisabledImage && $allowMultipleSelection) wire:click="toggleSelect('stockImage', '{{ $stockImage->id }}')"
								@elseif (!$isDisabledImage)
									wire:click="$set('selectedStockImages', ['{{ $stockImage->id }}'])" @endif
                                tabindex="{{ $isDisabledImage ? '-1' : '0' }}"
                                role="button"
                                @if ($isDisabledImage) aria-disabled="true" @endif
                            >
                                <div class="relative overflow-hidden rounded-lg">
                                    <img
                                        class="h-32 w-32 object-cover"
                                        src="{{ $stockImage->thumbnail }}"
                                        alt="{{ $stockImage->title }}"
                                        loading="lazy"
                                        onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik0xMiAxNkM4LjY4NjI5IDE2IDYgMTMuMzEzNyA2IDEwQzYgNi42ODYyOSA4LjY4NjI5IDQgMTIgNEMxNS4zMTM3IDQgMTggNi42ODYyOSAxOCAxMEMxOCAxMy4zMTM3IDE1LjMxMzcgMTYgMTIgMTZaIiBmaWxsPSIjOUI5QkEwIi8+CjxwYXRoIGQ9Ik0xMiAxMkMxMC44OTU0IDEyIDEwIDExLjEwNDYgMTAgMTBDMTAgOC44OTU0MyAxMC44OTU0IDggMTIgOEMxMy4xMDQ2IDggMTQgOC44OTU0MyAxNCAxMEMxNCAxMS4xMDQ2IDEzLjEwNDYgMTIgMTIgMTJaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K'"
                                    />

                                    {{-- Title overlay - matches images section style --}}
                                    <div
                                        class="{{ $isSelected ? 'opacity-100' : '' }} pointer-events-none absolute bottom-0 left-0 right-0 truncate bg-background p-1 text-xs text-heading-foreground opacity-0 transition-opacity duration-300 group-hover:opacity-100"
                                        style="pointer-events:none;"
                                    >
                                        {{ Str::limit($stockImage->title ?? 'Untitled', 30) }}
                                    </div>

                                    {{-- Selection indicator --}}
                                    @if ($isSelected)
                                        <span class="absolute end-1 top-1.5 flex items-center justify-center rounded-full bg-background p-2 shadow-lg">
                                            <x-tabler-check class="size-4" />
                                        </span>
                                    @endif
                                </div>
                            </div>
                        @empty
                            <div class="col-span-6 py-8 text-center text-gray-500">
                                <x-tabler-photo class="mx-auto mb-2 size-12 opacity-50" />
                                @if (empty($searchTerm))
                                    <p>@lang('Enter a search term to find stock images')</p>
                                @else
                                    <p>@lang('No stock images found for') "<strong>{{ $searchTerm }}</strong>"</p>
                                    <p class="mt-1 text-sm">@lang('Try adjusting your search term.')</p>
                                    <button
                                        class="mt-2 text-sm text-blue-500 underline hover:text-blue-700"
                                        wire:click="clearSearch"
                                    >
                                        @lang('Clear search')
                                    </button>
                                @endif
                            </div>
                        @endforelse
                    </div>

                    {{-- Loading spinner --}}
                    <div
                        class="flex items-center justify-center py-4"
                        wire:loading
                    >
                        <x-tabler-loader-2 class="size-6 animate-spin"></x-tabler-loader-2>
                    </div>
                </div>

                {{-- Insert button for selected stock images --}}
                @if (!empty($selectedStockImages) && count($selectedStockImages) > 0)
                    <div class="flex justify-end border-t border-gray-200 px-4 pb-4 pt-4">
                        <x-button
                            class="w-full p-2 opacity-90 hover:opacity-100"
                            wire:click="downloadAndInsertStockImages"
                            type="button"
                            size="sm"
                            variant="secondary"
                            wire:loading.attr="disabled"
                            wire:loading.class="opacity-50"
                        >
                            <span
                                wire:loading.remove
                                wire:target="downloadAndInsertStockImages"
                            >
                                @lang($allowMultipleSelection ? 'Download & Insert Selected Images' : 'Download & Insert Selected Image')
                                @if ($allowMultipleSelection)
                                    ({{ count($selectedStockImages) }})
                                @endif
                            </span>
                            <span
                                class="flex items-center"
                                wire:loading
                                wire:target="downloadAndInsertStockImages"
                            >
                                <svg
                                    class="-ml-1 mr-2 h-4 w-4 animate-spin text-white"
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                >
                                    <circle
                                        class="opacity-25"
                                        cx="12"
                                        cy="12"
                                        r="10"
                                        stroke="currentColor"
                                        stroke-width="4"
                                    ></circle>
                                    <path
                                        class="opacity-75"
                                        fill="currentColor"
                                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                    ></path>
                                </svg>
                                @lang('Downloading...')
                            </span>
                        </x-button>
                    </div>
                @endif
            @endif

            {{-- Stock Videos Section --}}
            @if ($activeFilter === 'Stock Videos')
                <div class="overflow-y-auto overscroll-contain p-4">
                    {{-- Search results info --}}
                    @if (!empty($searchTerm))
                        <div class="mb-4 text-sm text-gray-600">
                            @if ($stockVideos->count() > 0)
                                @lang('Found :count stock videos for', ['count' => $stockVideos->count()]) "<strong>{{ $searchTerm }}</strong>"
                            @else
                                @lang('No stock videos found for') "<strong>{{ $searchTerm }}</strong>"
                            @endif
                        </div>
                    @endif

                    <div
                        class="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6"
                        wire:loading.class="opacity-50"
                    >
                        @forelse ($stockVideos as $stockVideo)
                            @php
                                $isDisabledVideo = $this->isCardDisabled('video', $this->allowedTypes);
                                // Check if this stock video is selected
                                $isSelected = in_array((string) $stockVideo->id, $selectedStockVideos ?? [], true);
                            @endphp
                            <div
                                class="{{ $isSelected ? 'outline outline-[3px] outline-accent' : '' }} {{ $isDisabledVideo ? 'pointer-events-none opacity-30 bg-gray-50 border-gray-200 cursor-not-allowed filter grayscale' : 'cursor-pointer hover:bg-gray-100 hover:shadow-md hover:border-gray-300' }} group relative w-fit overflow-hidden rounded-lg border p-2.5 transition-all duration-200"
                                @if (!$isDisabledVideo && $allowMultipleSelection) wire:click="toggleSelect('stockVideo', '{{ $stockVideo->id }}')"
								@elseif (!$isDisabledVideo)
									wire:click="$set('selectedStockVideos', ['{{ $stockVideo->id }}'])" @endif
                                tabindex="{{ $isDisabledVideo ? '-1' : '0' }}"
                                role="button"
                                @if ($isDisabledVideo) aria-disabled="true" @endif
                                x-data="{ isPlaying: false, showControls: false, videoLoaded: false }"
                            >
                                <div class="relative overflow-hidden rounded-lg">
                                    {{-- Video thumbnail with hidden video player --}}
                                    @if ($stockVideo->thumbnail)
                                        <div class="relative">
                                            <img
                                                class="h-32 w-32 object-cover"
                                                src="{{ $stockVideo->thumbnail }}"
                                                alt="{{ $stockVideo->title }}"
                                                loading="lazy"
                                                x-show="!isPlaying"
                                                onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik0xMiAxNkM4LjY4NjI5IDE2IDYgMTMuMzEzNyA2IDEwQzYgNi42ODYyOSA4LjY4NjI5IDQgMTIgNEMxNS4zMTM3IDQgMTggNi42ODYyOSAxOCAxMEMxOCAxMy4zMTM3IDE1LjMxMzcgMTYgMTIgMTZaIiBmaWxsPSIjOUI5QkEwIi8+CjxwYXRoIGQ9Ik0xMiAxMkMxMC44OTU0IDEyIDEwIDExLjEwNDYgMTAgMTBDMTAgOC44OTU0MyAxMC44OTU0IDggMTIgOEMxMy4xMDQ2IDggMTQgOC44OTU0MyAxNCAxMEMxNCAxMS4xMDQ2IDEzLjEwNDYgMTIgMTIgMTJaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K'"
                                            />

                                            {{-- Hidden video player for stock videos --}}
                                            @if ($stockVideo->preview_url || $stockVideo->url)
                                                <video
                                                    class="absolute inset-0 h-32 w-32 object-cover"
                                                    x-show="isPlaying"
                                                    muted
                                                    preload="none"
                                                    x-ref="stockVideoPlayer"
                                                    @mouseenter="showControls = true"
                                                    @mouseleave="showControls = false"
                                                    @click.stop
                                                    @ended="isPlaying = false"
                                                    @play="isPlaying = true"
                                                    @pause="isPlaying = false"
                                                    @loadeddata="videoLoaded = true"
                                                >
                                                    <source
                                                        src="{{ $stockVideo->preview_url ?? $stockVideo->url }}"
                                                        type="video/mp4"
                                                    >
                                                </video>
                                            @endif
                                        </div>
                                    @else
                                        <div class="flex h-32 w-32 items-center justify-center bg-gray-200">
                                            <x-tabler-video class="size-8 text-gray-400" />
                                        </div>
                                    @endif

                                    {{-- Video Controls Overlay --}}
                                    @if ($stockVideo->preview_url || $stockVideo->url)
                                        <div
                                            class="absolute inset-0 flex items-center justify-center transition-opacity"
                                            x-show="showControls || !isPlaying"
                                            x-transition:enter="transition-opacity duration-200"
                                            x-transition:enter-start="opacity-0"
                                            x-transition:enter-end="opacity-100"
                                            x-transition:leave="transition-opacity duration-200"
                                            x-transition:leave-start="opacity-100"
                                            x-transition:leave-end="opacity-0"
                                            @mouseenter="showControls = true"
                                            @mouseleave="showControls = false"
                                        >
                                            <div class="rounded-full bg-black/50 p-2">
                                                <button
                                                    class="flex items-center justify-center text-white transition-colors hover:text-gray-300"
                                                    @click.stop="
														if (isPlaying) {
															$refs.stockVideoPlayer.pause();
														} else {
															$refs.stockVideoPlayer.play();
														}
													"
                                                >
                                                    <x-tabler-player-play
                                                        class="size-6"
                                                        x-show="!isPlaying"
                                                    />
                                                    <x-tabler-player-pause
                                                        class="size-6"
                                                        x-show="isPlaying"
                                                    />
                                                </button>
                                            </div>
                                        </div>
                                    @endif

                                    {{-- Duration badge --}}
                                    @if ($stockVideo->duration > 0)
                                        <div class="absolute bottom-2 left-2 rounded bg-black bg-opacity-70 px-1 py-0.5 text-xs text-white">
                                            {{ gmdate('i:s', $stockVideo->duration) }}
                                        </div>
                                    @endif

                                    {{-- Title overlay --}}
                                    <div
                                        class="{{ $isSelected ? 'opacity-100' : '' }} pointer-events-none absolute bottom-0 left-0 right-0 truncate bg-background p-1 text-xs text-heading-foreground opacity-0 transition-opacity duration-300 group-hover:opacity-100"
                                        style="pointer-events:none;"
                                    >
                                        {{ Str::limit($stockVideo->title ?? 'Untitled', 30) }}
                                    </div>

                                    {{-- Selection indicator --}}
                                    @if ($isSelected)
                                        <span class="absolute end-1 top-1.5 flex items-center justify-center rounded-full bg-background p-2 shadow-lg">
                                            <x-tabler-check class="size-4" />
                                        </span>
                                    @endif
                                </div>
                            </div>
                        @empty
                            <div class="col-span-6 py-8 text-center text-gray-500">
                                <x-tabler-video class="mx-auto mb-2 size-12 opacity-50" />
                                @if (empty($searchTerm))
                                    <p>@lang('Enter a search term to find stock videos')</p>
                                @else
                                    <p>@lang('No stock videos found for') "<strong>{{ $searchTerm }}</strong>"</p>
                                    <p class="mt-1 text-sm">@lang('Try adjusting your search term.')</p>
                                    <button
                                        class="mt-2 text-sm text-blue-500 underline hover:text-blue-700"
                                        wire:click="clearSearch"
                                    >
                                        @lang('Clear search')
                                    </button>
                                @endif
                            </div>
                        @endforelse
                    </div>

                    {{-- Loading spinner --}}
                    <div
                        class="flex items-center justify-center py-4"
                        wire:loading
                    >
                        <x-tabler-loader-2 class="size-6 animate-spin"></x-tabler-loader-2>
                    </div>
                </div>

                {{-- Insert button for selected stock videos --}}
                @if (!empty($selectedStockVideos) && count($selectedStockVideos) > 0)
                    <div class="flex justify-end border-t border-gray-200 px-4 pb-4 pt-4">
                        <x-button
                            class="w-full p-2 opacity-90 hover:opacity-100"
                            wire:click="downloadAndInsertStockVideos"
                            type="button"
                            size="sm"
                            variant="secondary"
                            wire:loading.attr="disabled"
                            wire:loading.class="opacity-50"
                        >
                            <span
                                wire:loading.remove
                                wire:target="downloadAndInsertStockVideos"
                            >
                                @lang($allowMultipleSelection ? 'Download & Insert Selected Videos' : 'Download & Insert Selected Video')
                                @if ($allowMultipleSelection)
                                    ({{ count($selectedStockVideos) }})
                                @endif
                            </span>
                            <span
                                class="flex items-center"
                                wire:loading
                                wire:target="downloadAndInsertStockVideos"
                            >
                                <svg
                                    class="-ml-1 mr-2 h-4 w-4 animate-spin text-white"
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                >
                                    <circle
                                        class="opacity-25"
                                        cx="12"
                                        cy="12"
                                        r="10"
                                        stroke="currentColor"
                                        stroke-width="4"
                                    ></circle>
                                    <path
                                        class="opacity-75"
                                        fill="currentColor"
                                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                    ></path>
                                </svg>
                                @lang('Downloading...')
                            </span>
                        </x-button>
                    </div>
                @endif
            @endif
        </x-card>
    </div>
</div>

<script>
    // Enhanced file upload handling
    document.addEventListener('livewire:init', () => {
        Livewire.on('uploadComplete', (data) => {
            //console.log('Upload completed:', data);
        });

        // Add global file input change listener for immediate feedback
        document.addEventListener('change', function(e) {
            if (e.target.matches('input[type="file"][data-exclude-media-manager]')) {
                const files = Array.from(e.target.files);
                if (files.length > 0) {
                    // Show immediate feedback for file selection
                    const fileNames = files.map(f => f.name).join(', ');
                    //console.log(`Selected files: ${fileNames}`);

                    // Check for large files and show warning
                    const largeFiles = files.filter(f => f.size > 10 * 1024 * 1024); // 10MB+
                    if (largeFiles.length > 0) {
                        // Create a temporary notification for large file processing
                        const tempNotification = document.createElement('div');
                        tempNotification.className = 'fixed top-4 right-4 z-[999999] px-4 py-3 rounded-lg shadow-lg text-white bg-blue-500';
                        tempNotification.innerHTML = `
							<div class="flex items-center gap-2">
								<div class="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
								<span>@lang('Processing large files, please wait...')</span>
							</div>
						`;
                        document.body.appendChild(tempNotification);

                        // Remove after file processing should be complete
                        setTimeout(() => {
                            if (tempNotification.parentNode) {
                                tempNotification.remove();
                            }
                        }, 1000);
                    }
                }
            }
        });

        Alpine.data('infiniteScroll', () => ({
            observer: null,
            mutationTimeout: null,

            init() {
                this.$nextTick(() => {
                    this.setupIntersectionObserver();
                });

                // Listen for Livewire updates and reinitialize observer
                this.$wire.on('searchUpdated', () => {
                    this.$nextTick(() => {
                        this.setupIntersectionObserver();
                    });
                });

                // Listen for upload completion
                this.$wire.on('uploadComplete', (data) => {
                    // Show success notification
                    if (data.message) {
                        this.showNotification(data.message, 'success');
                    }
                });

                // Listen for file processing events
                this.$wire.on('fileProcessingStarted', () => {
                    //console.log('File processing started');
                });

                this.$wire.on('fileProcessingCompleted', (data) => {
                    //console.log('File processing completed', data);
                    if (data.validFileCount > 0) {
                        this.showNotification(`${data.validFileCount} file(s) ready for upload`, 'info');
                    }
                    if (data.hasErrors) {
                        this.showNotification('Some files had validation errors', 'warning');
                    }
                });

                // Listen for upload progress clearing
                this.$wire.on('clearUploadProgress', () => {
                    setTimeout(() => {
                        this.$wire.uploadProgress = '';
                    }, 3000);
                });

                // Also listen for DOM updates via MutationObserver as fallback
                this.setupMutationObserver();
            },

            setupIntersectionObserver() {
                // Disconnect existing observer
                if (this.observer) {
                    this.observer.disconnect();
                }

                const loadTrigger = this.$refs.loadTrigger;
                if (!loadTrigger) {
                    // If no load trigger exists, try again after a short delay
                    setTimeout(() => {
                        this.setupIntersectionObserver();
                    }, 100);
                    return;
                }

                this.observer = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            // Add a small debounce to prevent multiple rapid calls
                            if (!this.isLoading) {
                                this.$wire.loadMore();
                            }
                        }
                    });
                }, {
                    root: this.$refs.scrollContainer,
                    threshold: 0.1,
                    rootMargin: '50px'
                });

                this.observer.observe(loadTrigger);
            },

            setupMutationObserver() {
                // Watch for changes in the scroll container
                const mutationObserver = new MutationObserver(() => {
                    // Debounce the observer setup
                    clearTimeout(this.mutationTimeout);
                    this.mutationTimeout = setTimeout(() => {
                        this.setupIntersectionObserver();
                    }, 100);
                });

                if (this.$refs.scrollContainer) {
                    mutationObserver.observe(this.$refs.scrollContainer, {
                        childList: true,
                        subtree: true
                    });
                }
            },

            showNotification(message, type = 'info') {
                // Enhanced notification system with better styling
                const notification = document.createElement('div');
                notification.className = `fixed top-4 right-4 z-[999999] px-4 py-3 rounded-lg shadow-lg text-white transform transition-all duration-300 translate-x-full opacity-0 ${
					type === 'success' ? 'bg-green-500' :
						type === 'error' ? 'bg-red-500' :
							type === 'warning' ? 'bg-yellow-500' :
								'bg-blue-500'
				}`;
                notification.textContent = message;

                document.body.appendChild(notification);

                // Animate in
                setTimeout(() => {
                    notification.style.transform = 'translateX(0)';
                    notification.style.opacity = '1';
                }, 100);

                // Animate out and remove
                setTimeout(() => {
                    notification.style.transform = 'translateX(full)';
                    notification.style.opacity = '0';
                    setTimeout(() => {
                        if (notification.parentNode) {
                            notification.remove();
                        }
                    }, 300);
                }, 4000);
            },

            destroy() {
                if (this.observer) {
                    this.observer.disconnect();
                }
                if (this.mutationTimeout) {
                    clearTimeout(this.mutationTimeout);
                }
            }
        }))

        Alpine.data('mediaManagerUploadArea', () => ({
            dragover: false,
            fileProcessing: false,
            handleDrop(e) {
                e.preventDefault();
                    this.dragover = false;
                    const files = Array.from(e.dataTransfer.files);
                    if (!files.length) return;

                    // Show processing state immediately
                    this.fileProcessing = true;

                    // Get reference to hidden file input
                    const fileInput = this.$refs.fileInput;
                    if (!fileInput) {
                        console.error('File input not found');
                        this.fileProcessing = false;
                        return;
                    }

                    try {
                        // Create DataTransfer object to simulate input change
                        const dt = new DataTransfer();
                        files.forEach(file => dt.items.add(file));

                        // Set files to the input
                        fileInput.files = dt.files;

                        // Trigger the input change event to make Livewire detect the change
                        fileInput.dispatchEvent(new Event('change', {
                            bubbles: true
                        }));
                    } catch (error) {
                        console.error('Error handling file drop:', error);
                        this.fileProcessing = false;
                    }
            },
            handleFileSelect() {
				// Show processing state immediately when files are selected
				this.fileProcessing = true;

				// Set a timeout to hide processing state if no files are shown
				setTimeout(() => {
					// Check if upload is still in progress
					if (!this.$wire.uploadingFiles || this.$wire.uploadingFiles.length === 0) {
						this.fileProcessing = false;
					}
				}, 10000); // 10 seconds timeout
            },
            handleDragOver(e) {
                e.preventDefault();
                this.dragover = true;
            },
            handleDragLeave(e) {
                e.preventDefault();
                // Only set dragover to false if we're actually leaving the drop area
                const rect = this.$el.getBoundingClientRect();
                const x = e.clientX;
                const y = e.clientY;

                if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
                    this.dragover = false;
                }
            }
        }))
    });
</script>
