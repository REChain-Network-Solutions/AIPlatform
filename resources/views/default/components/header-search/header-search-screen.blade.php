@php
    $style = $attributes->get('style') ?? setting('default_search', 'compact');
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
        'pending': pending
    }"
>
    <div class="{{ @twMerge('header-search-input-wrap relative w-full', $attributes->get('class:input-wrap')) }}">
        @if ($showIcon)
            <x-tabler-search
                class="{{ @twMerge('header-search-icon pointer-events-none absolute start-3 top-1/2 z-10 w-5 -translate-y-1/2 opacity-75', $attributes->get('class:icon')) }}"
                stroke-width="1.5"
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

        <x-forms.input
            :class="@twMerge('header-search-input border-none bg-heading-foreground/5 transition-colors placeholder-shown:text-ellipsis', $attributes->get('class:input'))"
            :container-class="@twMerge('peer', $attributes->get('class:input-container'))"
            type="text"
            @click.prevent="toggleModal(true)"
            onkeydown="return event.key != 'Enter';"
            placeholder="{{ __('Search for documents, templates, tools and more') }}"
            :x-ref="$attributes->has('x-ref') ? $attributes->get('x-ref') : null"
            x-model="searchTerm"
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

        @if ($showLoader)
            <span
                class="{{ @twMerge('header-search-loader absolute end-12 top-1/2 -translate-y-1/2', $attributes->get('class:loader')) }}"
                x-cloak
                x-show="isSearching"
            >
                <x-tabler-loader-2
                    class="animate-spin"
                    stroke-width="1.5"
                    role="status"
                />
            </span>
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

    @if ($style === 'compact')
        @include('components.header-search.header-search-results')
    @endif

    @if ($style === 'modern')
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
                    class="lqd-modal-content relative z-[100] max-h-[95vh] min-w-[clamp(250px,760px,90vw)] max-w-[min(calc(100%-2rem),630px)] overflow-y-auto overscroll-contain rounded-xl bg-background shadow-2xl shadow-black/10 group-[.modal-open]/header-search-modal:motion-scale-in-[0.98] group-[.modal-open]/header-search-modal:motion-opacity-in-[0%] group-[.modal-open]/header-search-modal:motion-duration-200">
                    {{-- Search input --}}
                    <div class="relative flex items-center justify-between border-b">
                        <div class="relative flex w-full">
                            <x-tabler-search
                                class="pointer-events-none absolute start-4 top-1/2 z-10 size-5 -translate-y-1/2 text-heading-foreground sm:start-5"
                                stroke-width="1.5"
                            />
                            <input
                                class="header-search-input h-[70px] w-full bg-transparent ps-12 text-heading-foreground outline-none transition-colors placeholder:text-foreground placeholder-shown:text-ellipsis max-lg:rounded-md max-sm:text-base sm:ps-14"
                                type="text"
                                onkeydown="return event.key != 'Enter';"
                                placeholder="{{ __('Search for anything...') }}"
                                x-model="searchTerm"
                                @input="handleSearch"
                                @focus="handleFocus"
                                @blur="handleBlur"
                            />

                            <span
                                class="{{ @twMerge('header-search-modal-loader absolute end-5 top-1/2 -translate-y-1/2', $attributes->get('class:modal-loader')) }}"
                                x-cloak
                                x-show="isSearching"
                                x-transition
                            >
                                <x-tabler-loader-2
                                    class="animate-spin"
                                    stroke-width="1.5"
                                    role="status"
                                />
                            </span>
                        </div>

                        <div
                            class="pointer-events-none absolute end-5 top-1/2 flex -translate-y-1/2 whitespace-nowrap max-md:hidden"
                            x-show="!isSearching"
                            x-transition
                        >
                            <span class="text-3xs font-normal text-foreground/50 text-opacity-50">
                                {{ 'Templates, Documents, Bots and more' }}
                            </span>
                        </div>
                    </div>

                    @include('components.header-search.header-search-results')
                </div>
            </div>
        </template>
    @endif
</form>
