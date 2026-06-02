@php
    $base_class = 'lqd-selectbox';
    $selectboxId = 'lqd-selectbox-' . uniqid();
    $optionsScriptId = $selectboxId . '-options';
    $optionsForJs = collect($options)
        ->map(
            fn($o) => [
                'value' => (string) ($o['value'] ?? ''),
                'label' => (string) ($o['label'] ?? ''),
            ],
        )
        ->values();
    $hasArrayOptions = $optionsForJs->isNotEmpty();
@endphp

<div
    {{ $attributes->twMerge($base_class) }}
    x-data="liquidSelectBox({
        value: {{ json_encode($value) }},
        optionsScriptId: {{ $hasArrayOptions ? json_encode($optionsScriptId) : 'null' }},
        placeholder: {{ json_encode($placeholder) }}
    })"
>
    @if ($label)
        <label class="lqd-selectbox-label lqd-input-label mb-3 flex cursor-pointer items-center gap-2 text-2xs font-medium leading-none text-label">
            {{ $label }}
            @if ($tooltip)
                <x-info-tooltip>{{ $tooltip }}</x-info-tooltip>
            @endif
        </label>
    @endif

    @if ($name)
        <input
            type="hidden"
            name="{{ $name }}"
            value="{{ $value ?? '' }}"
            :value="selected ?? ''"
        />
    @endif

    <x-dropdown.dropdown
        class="lqd-selectbox-dropdown w-full grow"
        class:dropdown="lqd-selectbox-dropdown-dropdown max-h-[min(75vh,500px)] w-[min(calc(100vw-30px),400px)] overflow-y-auto bg-transparent px-2 pb-4 pt-0 shadow-[0_10px_15px_-3px_hsl(0_0%_0%/10%)] before:absolute before:inset-0 before:rounded-[inherit] before:bg-dropdown-background/90 before:backdrop-blur-xl"
        triggerType="click"
        offsetY="0.25rem"
    >
        <x-slot:trigger
            class="lqd-selectbox-trigger h-11 min-h-10 w-full justify-between overflow-hidden !rounded-input bg-background text-start font-normal capitalize hover:shadow-none sm:text-2xs"
            variant="outline"
            type="button"
        >
            <span
                class="w-full grow truncate"
                x-text="activeLabel"
            >{{ $activeLabel ?? $placeholder }}</span>
            <x-tabler-chevron-down class="size-4 shrink-0" />
        </x-slot:trigger>

        <x-slot:dropdown>
            <div x-trap="open">
                <div class="sticky top-0 z-1 z-10 -mx-2 mb-2 mt-1 px-2 pt-2 backdrop-blur-lg">
                    <x-forms.input
                        class="lqd-selectbox-search-input mb-3 shadow-md shadow-black/5"
                        type="search"
                        placeholder="{{ __('Search...') }}"
                        x-model="query"
                    />
                </div>

                <div class="relative z-1 space-y-0.5">
                    <template
                        x-for="option in visibleOptions"
                        :key="option.value"
                    >
                        <button
                            class="lqd-selectbox-option flex w-full items-center justify-normal gap-2.5 overflow-hidden rounded-lg px-4 py-2.5 text-start text-2xs font-medium text-heading-foreground transition hover:bg-foreground/5"
                            type="button"
                            :class="{ 'bg-foreground/5': selected === option.value }"
                            @click.prevent="select(option.value); toggle('collapse')"
                        >
                            <span
                                class="w-full truncate"
                                x-text="option.label"
                            ></span>
                            <template x-if="selected === option.value">
                                <x-tabler-check class="ms-auto size-5 shrink-0" />
                            </template>
                        </button>
                    </template>

                    {{ $slot }}

                    <div
                        x-ref="loadMoreSentinel"
                        aria-hidden="true"
                    ></div>

                    <p
                        class="lqd-selectbox-no-results m-0 px-4 py-6 text-center text-2xs opacity-60"
                        x-show="query && !hasMatches"
                        x-cloak
                    >
                        {{ __('No options match your search.') }}
                    </p>
                </div>
            </div>
        </x-slot:dropdown>
    </x-dropdown.dropdown>
</div>

@if ($hasArrayOptions)
    @push('script')
        <script
            type="application/json"
            id="{{ $optionsScriptId }}"
        >@json($optionsForJs)</script>
    @endpush
@endif
