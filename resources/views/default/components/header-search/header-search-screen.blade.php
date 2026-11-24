@php
    $style = $attributes->get('style') ?? 'modern';
@endphp

<form
    @class([
        @twMerge(
            'header-search relative transition-all group/header-search',
            $attributes->get('class')),
        'header-search-style-' . $style,
    ])
    x-data="liquidHeaderSearch"
    @keyup.escape.window="toggleModal(false)"
    :class="{
        'open': isSearching || doneSearching || pending,
        'is-searching': isSearching,
        'done-searching': doneSearching,
        'pending': pending,
        'ai-mode': aiMode
    }"
>
    <div class="{{ @twMerge('header-search-input-wrap relative w-full', $attributes->get('class:input-wrap')) }}">
        @if ($showIcon)
            <x-tabler-search
                class="{{ @twMerge('header-search-icon pointer-events-none absolute start-3 top-1/2 z-10 w-5 -translate-y-1/2 opacity-75', $attributes->get('class:icon')) }}"
                stroke-width="1.5"
                x-show="!aiMode"
            />
        @endif

        @if ($outlineGlow)
            <div
                class="{{ @twMerge('header-search-border pointer-events-none absolute -inset-1 overflow-hidden rounded-[calc(var(--input-rounded)*var(--input-rounded-multiplier)+0.25rem)] bg-heading-foreground/5', $attributes->get('class:input-glow-wrap')) }}">
                <div class="header-search-border-play absolute left-1/2 top-1/2 aspect-square min-h-full min-w-full -translate-x-1/2 -translate-y-1/2 rounded-[inherit]">
                    <div class="header-search-border-play-inner absolute min-h-full min-w-full opacity-0"></div>
                </div>
            </div>
        @endif

        <span
            class="header-search-ai-indicator pointer-events-none absolute start-4 top-1/2 z-10 inline-grid size-6 shrink-0 -translate-y-1/2 place-items-center rounded-full bg-gradient-to-r from-gradient-from via-gradient-via to-gradient-to text-white before:absolute before:-z-10 before:size-[125%] before:animate-spin-grow before:opacity-25 before:blur-md before:[background:inherit]"
            x-show="aiMode"
            x-cloak
        >
            <svg
                class="size-3.5"
                width="19"
                height="18"
                viewBox="0 0 19 18"
                fill="currentColor"
                xmlns="http://www.w3.org/2000/svg"
            >
                <path
                    d="M6.17912 14.5715L6.59586 14.4869C6.71257 14.4635 6.81757 14.4004 6.89301 14.3083C6.96846 14.2162 7.00968 14.1009 7.00968 13.9818C7.00968 13.8628 6.96846 13.7475 6.89301 13.6554C6.81757 13.5633 6.71257 13.5002 6.59586 13.4768L6.17912 13.3922C5.66528 13.288 5.19354 13.0346 4.82279 12.6639C4.45204 12.2931 4.19873 11.8214 4.09449 11.3076L4.00987 10.8908C3.98648 10.7741 3.92339 10.6691 3.83132 10.5937C3.73924 10.5182 3.62388 10.477 3.50485 10.477C3.38581 10.477 3.27045 10.5182 3.17838 10.5937C3.08631 10.6691 3.0232 10.7741 2.99981 10.8908L2.91519 11.3076C2.81095 11.8214 2.55764 12.2931 2.18689 12.6639C1.81615 13.0346 1.3444 13.288 0.830556 13.3922L0.413842 13.4768C0.297129 13.5002 0.192112 13.5633 0.116666 13.6554C0.04122 13.7475 0 13.8628 0 13.9818C0 14.1009 0.04122 14.2162 0.116666 14.3083C0.192112 14.4004 0.297129 14.4635 0.413842 14.4869L0.830556 14.5715C1.3444 14.6758 1.81615 14.9291 2.18689 15.2998C2.55764 15.6706 2.81095 16.1423 2.91519 16.6561L2.99981 17.0729C3.0232 17.1896 3.08631 17.2946 3.17838 17.37C3.27045 17.4455 3.38581 17.4867 3.50485 17.4867C3.62388 17.4867 3.73924 17.4455 3.83132 17.37C3.92339 17.2946 3.98648 17.1896 4.00987 17.0729L4.09449 16.6561C4.19873 16.1423 4.45204 15.6706 4.82279 15.2998C5.19354 14.9291 5.66528 14.6758 6.17912 14.5715Z"
                />
                <path
                    d="M16.2815 7.02509L17.8151 6.71415C17.9682 6.68285 18.1057 6.59965 18.2045 6.47862C18.3033 6.3576 18.3572 6.20616 18.3572 6.04993C18.3572 5.89371 18.3033 5.74229 18.2045 5.62126C18.1057 5.50023 17.9682 5.41704 17.8151 5.38574L16.2815 5.07478C15.5427 4.92485 14.8644 4.56061 14.3314 4.02754C13.7983 3.49447 13.434 2.81619 13.2841 2.07737L12.9732 0.543773C12.9424 0.390412 12.8595 0.252432 12.7385 0.1533C12.6175 0.0541683 12.4659 0 12.3095 0C12.1531 0 12.0015 0.0541683 11.8805 0.1533C11.7595 0.252432 11.6766 0.390412 11.6458 0.543773L11.3348 2.07737C11.185 2.81625 10.8209 3.4946 10.2878 4.02769C9.75469 4.56078 9.07634 4.92498 8.33746 5.07478L6.80385 5.38574C6.65079 5.41704 6.51323 5.50023 6.41445 5.62126C6.31567 5.74229 6.26172 5.89371 6.26172 6.04993C6.26172 6.20616 6.31567 6.3576 6.41445 6.47862C6.51323 6.59965 6.65079 6.68285 6.80385 6.71415L8.33746 7.02509C9.07634 7.17489 9.75469 7.53909 10.2878 8.07218C10.8209 8.60527 11.185 9.28364 11.3348 10.0225L11.6458 11.5561C11.6766 11.7095 11.7595 11.8474 11.8805 11.9466C12.0015 12.0457 12.1531 12.0999 12.3095 12.0999C12.4659 12.0999 12.6175 12.0457 12.7385 11.9466C12.8595 11.8474 12.9424 11.7095 12.9732 11.5561L13.2841 10.0225C13.434 9.2837 13.7983 8.60541 14.3314 8.07234C14.8644 7.53927 15.5427 7.17502 16.2815 7.02509Z"
                />
            </svg>
            <span class="sr-only">
                {{ __('AI Assistant') }}
            </span>
        </span>

        <x-forms.input
            :class="@twMerge('header-search-input border-none bg-heading-foreground/5 transition-colors placeholder-shown:text-ellipsis', $attributes->get('class:input'))"
            :container-class="@twMerge('peer', $attributes->get('class:input-container'))"
            type="text"
            @click.prevent="toggleModal(true)"
            onkeydown="return event.key != 'Enter';"
            :x-ref="$attributes->has('x-ref') ? $attributes->get('x-ref') : null"
            disabled
            x-init="$el.disabled = false"
            x-model="searchTerm"
            x-bind:class="aiMode ? 'ps-12' : ''"
            x-bind:placeholder="aiMode ? '{{ __('Ask me anything…') }}' : '{{ __('Search anything or type /ai') }}'"
            placeholder="{{ __('Search anything or type /ai') }}"
            @input="handleSearch"
            @focus="handleFocus"
            @blur="handleBlur"
        />

        @if ($showKbd)
            <kbd
                class="{{ @twMerge('header-search-kbd peer-focus-within:scale-70 pointer-events-none absolute end-3 top-1/2 z-10 inline-block -translate-y-1/2 rounded-full bg-background px-2 py-1 text-3xs leading-none opacity-0 transition-all group-[.is-searching]/header-search:invisible group-[.is-searching]/header-search:opacity-0 peer-focus-within:invisible peer-focus-within:opacity-0', $attributes->get('class:kbd')) }}">
                <span
                    class="search-shortcut-key"
                    x-text="shortcutKey"
                ></span> + K
            </kbd>
        @endif

        @if ($showArrow)
            <span
                class="{{ @twMerge('header-search-arrow pointer-events-none absolute end-3 top-1/2 -translate-x-2 -translate-y-1/2 opacity-0 transition-all peer-focus-within:translate-x-0 peer-focus-within:opacity-100 rtl:-scale-x-100', $attributes->get('class:arrow')) }}"
                x-show="!isSearching && !doneSearching"
            >
                <x-tabler-chevron-right class="size-5" />
            </span>
        @endif
    </div>

    <template x-teleport="body">
        <div
            class="lqd-modal-modal header-search-modal group/header-search-modal fixed inset-0 z-[999] flex items-center justify-center overflow-y-auto overscroll-contain"
            x-show="modalOpen"
            x-cloak
            x-trap="modalOpen"
            :class="{ 'modal-open': modalOpen }"
        >
            <div
                class="lqd-modal-backdrop fixed inset-0 bg-black/30 group-[.modal-open]/header-search-modal:motion-preset-fade group-[.modal-open]/header-search-modal:motion-duration-200"
                @click="toggleModal(false)"
            >
            </div>
            <div
                class="lqd-modal-content relative z-[100] flex h-[560px] max-h-[90vh] min-w-[clamp(250px,760px,90vw)] max-w-[min(calc(100%-2rem),630px)] flex-col overflow-hidden overscroll-contain rounded-xl bg-background shadow-2xl shadow-black/10 group-[.modal-open]/header-search-modal:motion-scale-in-[0.98] group-[.modal-open]/header-search-modal:motion-opacity-in-[0%] group-[.modal-open]/header-search-modal:motion-duration-200">
                {{-- Search input --}}
                <div class="relative flex shrink-0 items-center border-b">
                    <x-tabler-search
                        class="pointer-events-none absolute start-4 top-1/2 z-10 size-5 -translate-y-1/2 text-heading-foreground sm:start-5"
                        stroke-width="1.5"
                        x-show="!aiMode"
                    />
                    <div
                        class="flex h-[70px] w-full items-center gap-2.5 pe-28 md:pe-36"
                        :class="aiMode ? 'ps-4 sm:ps-5' : 'ps-12 sm:ps-14'"
                    >
                        <span
                            class="header-search-ai-indicator pointer-events-none relative inline-grid size-6 shrink-0 place-items-center rounded-full bg-gradient-to-r from-gradient-from via-gradient-via to-gradient-to text-white before:absolute before:-z-10 before:size-[125%] before:animate-spin-grow before:opacity-25 before:blur-md before:[background:inherit]"
                            x-show="aiMode"
                            x-cloak
                        >
                            <svg
                                class="size-3.5"
                                width="19"
                                height="18"
                                viewBox="0 0 19 18"
                                fill="currentColor"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    d="M6.17912 14.5715L6.59586 14.4869C6.71257 14.4635 6.81757 14.4004 6.89301 14.3083C6.96846 14.2162 7.00968 14.1009 7.00968 13.9818C7.00968 13.8628 6.96846 13.7475 6.89301 13.6554C6.81757 13.5633 6.71257 13.5002 6.59586 13.4768L6.17912 13.3922C5.66528 13.288 5.19354 13.0346 4.82279 12.6639C4.45204 12.2931 4.19873 11.8214 4.09449 11.3076L4.00987 10.8908C3.98648 10.7741 3.92339 10.6691 3.83132 10.5937C3.73924 10.5182 3.62388 10.477 3.50485 10.477C3.38581 10.477 3.27045 10.5182 3.17838 10.5937C3.08631 10.6691 3.0232 10.7741 2.99981 10.8908L2.91519 11.3076C2.81095 11.8214 2.55764 12.2931 2.18689 12.6639C1.81615 13.0346 1.3444 13.288 0.830556 13.3922L0.413842 13.4768C0.297129 13.5002 0.192112 13.5633 0.116666 13.6554C0.04122 13.7475 0 13.8628 0 13.9818C0 14.1009 0.04122 14.2162 0.116666 14.3083C0.192112 14.4004 0.297129 14.4635 0.413842 14.4869L0.830556 14.5715C1.3444 14.6758 1.81615 14.9291 2.18689 15.2998C2.55764 15.6706 2.81095 16.1423 2.91519 16.6561L2.99981 17.0729C3.0232 17.1896 3.08631 17.2946 3.17838 17.37C3.27045 17.4455 3.38581 17.4867 3.50485 17.4867C3.62388 17.4867 3.73924 17.4455 3.83132 17.37C3.92339 17.2946 3.98648 17.1896 4.00987 17.0729L4.09449 16.6561C4.19873 16.1423 4.45204 15.6706 4.82279 15.2998C5.19354 14.9291 5.66528 14.6758 6.17912 14.5715Z"
                                />
                                <path
                                    d="M16.2815 7.02509L17.8151 6.71415C17.9682 6.68285 18.1057 6.59965 18.2045 6.47862C18.3033 6.3576 18.3572 6.20616 18.3572 6.04993C18.3572 5.89371 18.3033 5.74229 18.2045 5.62126C18.1057 5.50023 17.9682 5.41704 17.8151 5.38574L16.2815 5.07478C15.5427 4.92485 14.8644 4.56061 14.3314 4.02754C13.7983 3.49447 13.434 2.81619 13.2841 2.07737L12.9732 0.543773C12.9424 0.390412 12.8595 0.252432 12.7385 0.1533C12.6175 0.0541683 12.4659 0 12.3095 0C12.1531 0 12.0015 0.0541683 11.8805 0.1533C11.7595 0.252432 11.6766 0.390412 11.6458 0.543773L11.3348 2.07737C11.185 2.81625 10.8209 3.4946 10.2878 4.02769C9.75469 4.56078 9.07634 4.92498 8.33746 5.07478L6.80385 5.38574C6.65079 5.41704 6.51323 5.50023 6.41445 5.62126C6.31567 5.74229 6.26172 5.89371 6.26172 6.04993C6.26172 6.20616 6.31567 6.3576 6.41445 6.47862C6.51323 6.59965 6.65079 6.68285 6.80385 6.71415L8.33746 7.02509C9.07634 7.17489 9.75469 7.53909 10.2878 8.07218C10.8209 8.60527 11.185 9.28364 11.3348 10.0225L11.6458 11.5561C11.6766 11.7095 11.7595 11.8474 11.8805 11.9466C12.0015 12.0457 12.1531 12.0999 12.3095 12.0999C12.4659 12.0999 12.6175 12.0457 12.7385 11.9466C12.8595 11.8474 12.9424 11.7095 12.9732 11.5561L13.2841 10.0225C13.434 9.2837 13.7983 8.60541 14.3314 8.07234C14.8644 7.53927 15.5427 7.17502 16.2815 7.02509Z"
                                />
                            </svg>
                            <span class="sr-only">
                                {{ __('AI Assistant') }}
                            </span>
                        </span>
                        <input
                            class="header-search-input h-full w-full min-w-0 bg-transparent text-heading-foreground outline-none transition-colors placeholder:text-foreground placeholder-shown:text-ellipsis max-sm:text-base"
                            type="text"
                            onkeydown="return event.key != 'Enter';"
                            :placeholder="aiMode ? '{{ __('Ask me anything…') }}' : '{{ __('Search anything or type /ai') }}'"
                            x-model="searchTerm"
                            @input="handleSearch"
                            @focus="handleFocus"
                            @blur="handleBlur"
                        />
                    </div>

                    {{-- Right controls --}}
                    <div class="absolute end-4 top-1/2 flex -translate-y-1/2 items-center gap-1.5">
                        <span
                            class="{{ @twMerge('header-search-modal-loader', $attributes->get('class:modal-loader')) }}"
                            x-cloak
                            x-show="isSearching"
                            x-transition
                        >
                            <x-tabler-loader-2
                                class="size-4 animate-spin"
                                stroke-width="1.5"
                                role="status"
                            />
                        </span>
                        <kbd
                            class="hidden items-center gap-0.5 rounded-lg border bg-heading-foreground/5 px-2 py-1 font-sans text-xs leading-none text-foreground/40 md:inline-flex"
                            x-show="!isSearching"
                            x-transition
                        >
                            <span x-text="shortcutKey === 'cmd' ? '⌘' : 'Ctrl'"></span>K
                        </kbd>
                        <button
                            class="flex size-7 items-center justify-center rounded-lg transition hover:bg-heading-foreground/10 hover:text-foreground/60"
                            type="button"
                            @click.prevent="toggleModal(false)"
                            aria-label="{{ __('Close search') }}"
                        >
                            <x-tabler-x class="size-4" />
                        </button>
                    </div>
                </div>

                {{-- Filter tabs --}}
                @php
                    $searchFilters = [
                        ['key' => 'all', 'label' => __('All'), 'icon' => 'tabler-apps'],
                        ['key' => 'navigation', 'label' => __('Navigation'), 'icon' => 'tabler-menu-deep'],
                        ['key' => 'templates', 'label' => __('Templates'), 'icon' => 'tabler-layout-2'],
                        ['key' => 'documents', 'label' => __('Documents'), 'icon' => 'tabler-file-text'],
                    ];
                @endphp
                <div class="scrollbar-none flex shrink-0 items-center gap-1 overflow-x-auto border-b px-4 py-2.5">
                    @foreach ($searchFilters as $filter)
                        <button
                            class="flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-colors"
                            type="button"
                            :class="activeFilter === '{{ $filter['key'] }}' ? 'bg-heading-foreground/5 text-heading-foreground' : 'text-foreground/50 hover:text-foreground/70'"
                            @click="activeFilter = '{{ $filter['key'] }}'; focusedIndex = -1"
                        >
                            <x-dynamic-component
                                class="size-3.5 shrink-0"
                                :component="$filter['icon']"
                                stroke-width="1.5"
                            />
                            {{ $filter['label'] }}
                        </button>
                    @endforeach
                </div>

                <div class="min-h-0 flex-1 overflow-hidden">
                    @include('components.header-search.header-search-results')
                </div>

                {{-- Footer shortcuts --}}
                <div class="flex shrink-0 items-center gap-5 border-t px-5 py-2.5 text-2xs">
                    <span class="flex items-center gap-1.5">
                        {{ __('Open') }}
                        <kbd class="inline-flex size-5 items-center justify-center rounded border bg-heading-foreground/5 font-mono text-foreground/50">↵</kbd>
                    </span>
                    <span class="flex items-center gap-1.5">
                        {{ __('Close') }}
                        <kbd class="inline-flex items-center justify-center px-1.5 py-0.5 font-mono text-foreground/30">Esc</kbd>
                    </span>
                    <span class="flex items-center gap-1.5">
                        {{ __('Navigate') }}
                        <kbd class="inline-flex items-center justify-center px-1.5 py-0.5 font-mono text-foreground/30">↓ ↑</kbd>
                    </span>
                </div>
            </div>
        </div>
    </template>
</form>
