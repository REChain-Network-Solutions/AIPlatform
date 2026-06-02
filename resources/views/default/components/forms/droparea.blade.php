@php
    $base_class = 'lqd-droparea';
@endphp

<div
    {{ $attributes->twMerge($base_class) }}
    x-data="liquidDropArea({
        multiple: {{ $multiple ? 'true' : 'false' }},
        accept: {{ json_encode($accept ?? '') }},
        maxSizeMb: {{ $maxSizeMb !== null ? (int) $maxSizeMb : 'null' }}
    })"
>
    @if ($label)
        <label class="lqd-droparea-label lqd-input-label mb-3 flex cursor-pointer items-center gap-2 text-2xs font-medium leading-none text-label">
            {{ $label }}
            @if ($tooltip)
                <x-info-tooltip>{{ $tooltip }}</x-info-tooltip>
            @endif
        </label>
    @endif

    <input
        class="hidden"
        type="file"
        name="{{ $multiple ? $name . '[]' : $name }}"
        x-ref="fileInput"
        accept="{{ $accept }}"
        @if ($multiple) multiple @endif
        @change="handleFileSelect($event)"
    />

    <div
        @class([
            'lqd-droparea-surface group/droparea cursor-pointer border border-dashed border-foreground/10 p-6 text-center transition sm:p-8 [&.drag-over]:border-primary [&.drag-over]:bg-primary/5',
            'rounded-[10px]' => $style === 'compact',
            'rounded-[21px]' => $style === 'extended',
        ])
        :class="{ 'drag-over': dragOver }"
        @click="$refs.fileInput.click()"
        @dragover.prevent="dragOver = true"
        @dragleave.prevent="dragOver = false"
        @drop.prevent="handleFileDrop($event)"
    >
        <div
            class="lqd-droparea-empty"
            x-show="files.length === 0"
        >
            @if (filled($decorator))
                {!! $decorator !!}
            @endif

            <svg
                class="mx-auto mb-2.5 opacity-25 transition group-hover/droparea:-translate-y-1 group-hover/droparea:opacity-50"
                width="38"
                height="38"
                viewBox="0 0 38 38"
                fill="currentColor"
                xmlns="http://www.w3.org/2000/svg"
            >
                <path
                    d="M32.4073 32.4839C28.7298 36.1613 24.2608 38 19 38C13.7392 38 9.24462 36.1613 5.51613 32.4839C1.83871 28.7554 0 24.2608 0 19C0 13.7392 1.83871 9.27016 5.51613 5.59274C9.24462 1.86425 13.7392 0 19 0C24.2608 0 28.7298 1.86425 32.4073 5.59274C36.1358 9.27016 38 13.7392 38 19C38 24.2608 36.1358 28.7554 32.4073 32.4839ZM29.8024 8.19758C26.8401 5.18414 23.2392 3.67742 19 3.67742C14.7608 3.67742 11.1344 5.18414 8.12097 8.19758C5.1586 11.1599 3.67742 14.7608 3.67742 19C3.67742 23.2392 5.1586 26.8656 8.12097 29.879C11.1344 32.8414 14.7608 34.3226 19 34.3226C23.2392 34.3226 26.8401 32.8414 29.8024 29.879C32.8159 26.8656 34.3226 23.2392 34.3226 19C34.3226 14.7608 32.8159 11.1599 29.8024 8.19758ZM20.5323 28.8065H17.4677C16.8548 28.8065 16.5484 28.5 16.5484 27.8871V22C16.5484 20.3431 15.2052 19 13.5484 19H11.4153C11.0067 19 10.7258 18.8212 10.5726 18.4637C10.4194 18.0551 10.4704 17.7231 10.7258 17.4677L18.3871 9.80645C18.7957 9.39785 19.2043 9.39785 19.6129 9.80645L27.2742 17.4677C27.5296 17.7231 27.5806 18.0551 27.4274 18.4637C27.2742 18.8212 26.9933 19 26.5847 19H24.4516C22.7948 19 21.4516 20.3431 21.4516 22V27.8871C21.4516 28.5 21.1452 28.8065 20.5323 28.8065Z"
                ></path>
            </svg>

            <p class="mb-2 text-sm font-medium">
                {{ $placeholder ?? __('Drag and drop or click to browse') }}
            </p>

            @if ($style === 'compact')
                @if ($maxSizeMb)
                    <p class="m-0 text-4xs font-medium opacity-50">
                        {{ __('Max File Size: :size MB', ['size' => $maxSizeMb]) }}
                    </p>
                @endif
            @elseif($style === 'extended')
                <div class="mx-auto my-5 flex w-[min(100%,305px)] items-center gap-7">
                    <span class="inline-block h-px grow bg-current opacity-5"></span>
                    <span>
                        {{ __('or') }}
                    </span>
                    <span class="inline-block h-px grow bg-current opacity-5"></span>
                </div>

                <x-button
                    class="mx-auto mb-5 px-5.5 py-3 text-sm font-medium"
                    variant="outline"
                    @click.prevent="$refs.fileInput.click()"
                >
                    {{ __('Browse Files') }}
                </x-button>

                @if (filled($terms))
                    <ul class="flex w-full list-inside list-disc flex-col items-center text-[12px] text-foreground/50">
                        @foreach ($terms as $term)
                            <li>
                                {!! $term !!}
                            </li>
                        @endforeach
                    </ul>
                @else
                    @if ($maxSizeMb)
                        <ul class="flex w-full list-inside list-disc flex-col items-center text-[12px] text-foreground/50">
                            <li>
                                {{ __('Max File Size: :size MB', ['size' => $maxSizeMb]) }}
                            </li>
                        </ul>
                    @endif
                @endif
            @endif
        </div>

        <div
            class="lqd-droparea-previews"
            x-show="files.length > 0"
            x-cloak
        >
            <ul class="flex flex-wrap items-start justify-center gap-3">
                <template
                    x-for="(file, index) in files"
                    :key="file.name + '-' + file.size + '-' + file.lastModified"
                >
                    <li class="lqd-droparea-preview relative">
                        @isset($preview)
                            {{ $preview }}
                        @else
                            <div class="lqd-droparea-preview-default rounded-lg border bg-background p-2 text-start">
                                <template x-if="previewType(file) === 'image'">
                                    <img
                                        class="block max-h-32 w-auto rounded-md object-cover"
                                        :src="previewUrl(file)"
                                        :alt="file.name"
                                    />
                                </template>

                                <template x-if="previewType(file) === 'video'">
                                    <video
                                        class="block max-h-32 w-auto rounded-md"
                                        :src="previewUrl(file)"
                                        controls
                                        @click.stop
                                    ></video>
                                </template>

                                <template x-if="previewType(file) === 'audio'">
                                    <div class="flex items-center gap-2">
                                        <x-tabler-music class="size-5 shrink-0 opacity-60" />
                                        <audio
                                            class="h-8"
                                            :src="previewUrl(file)"
                                            controls
                                            @click.stop
                                        ></audio>
                                    </div>
                                </template>

                                <template x-if="previewType(file) === 'file'">
                                    <div class="flex items-center gap-2 text-2xs">
                                        <x-tabler-file class="size-5 shrink-0 opacity-60" />
                                        <div class="flex flex-col items-start">
                                            <span
                                                class="max-w-40 truncate font-medium"
                                                x-text="file.name"
                                            ></span>
                                            <span
                                                class="opacity-50"
                                                x-text="fileSize(file.size)"
                                            ></span>
                                        </div>
                                    </div>
                                </template>
                            </div>
                        @endisset

                        <button
                            class="lqd-droparea-remove absolute -end-2 -top-2 inline-grid size-7 place-items-center rounded-full bg-background text-foreground shadow-lg transition hover:scale-110 hover:bg-red-500 hover:text-white"
                            type="button"
                            :title="`{{ __('Remove') }} ${file.name}`"
                            @click.stop="removeFile(index)"
                        >
                            <x-tabler-x class="size-4" />
                        </button>
                    </li>
                </template>
            </ul>
        </div>

        <p
            class="lqd-droparea-error mt-3 text-2xs text-red-500"
            x-show="error"
            x-cloak
            x-text="error"
        ></p>
    </div>
</div>
