@php
    $container_class = ['header-search-results-container max-h-[min(440px,80vh)] overflow-y-auto rounded-xl bg-background shadow-xl shadow-black/5'];

    if ($style === 'compact') {
        $container_class[] = 'absolute top-full inset-x-0 z-50 mt-4';
    }

    $container_class[] = $attributes->get('class:results-container');
@endphp

<div
    class="{{ @twMerge($container_class, $attributes->get('class:results-container')) }}"
    @if ($style === 'compact') x-cloak
    x-transition
    x-show="modalOpen"
	@click.outside="toggleModal(false)" @endif
>
    <div
        class="header-search-results w-full"
        x-cloak
        x-show="doneSearching"
    >
        <h3 class="m-0 border-b py-4 ps-5 text-start text-base font-medium">
            {{ __('Search results') }}
        </h3>

        <!-- Search results here -->
        <div
            class="search-results-container"
            x-html="searchResults"
        ></div>
    </div>

    <div
        class="header-search-recents"
        x-cloak
        x-show="isSearching || pending"
    >
        <h3 class="m-0 w-full border-b px-5 py-4 text-start text-base font-medium">
            {{ __('Recent Search') }}
        </h3>

        <!-- Recent searchs -->
        <div
            class="recent-search-container flex flex-col"
            :class="{ 'no-result': recentSearchKeys.length === 0 }"
        >
            <h3 class="mb-2 hidden p-6 text-center font-medium text-heading-foreground">
                {{ __('There is no recent search.') }}
            </h3>

            <div class="flex flex-wrap gap-2 p-5">
                <template
                    x-for="key in recentSearchKeys"
                    :key="key.keyword || key"
                >
                    <a
                        class="flex cursor-pointer items-center gap-3 rounded-2xl border py-0.5 pe-1 ps-3 text-xs font-medium transition-all hover:border-primary hover:bg-primary hover:text-primary-foreground"
                        href="#"
                        @click.prevent="applyRecentSearch(key)"
                    >
                        <span
                            class="leading-none"
                            x-text="key.keyword || key"
                        ></span>

                        <span
                            class="inline-grid size-[18px] cursor-pointer place-items-center rounded-full transition hover:scale-110 hover:bg-background hover:text-foreground"
                            @click.prevent.stop="deleteRecentSearchKey(key)"
                        >
                            <x-tabler-x class="size-4 opacity-80" />
                        </span>
                    </a>
                </template>
            </div>
        </div>

        <x-button
            class="w-full justify-between rounded-none border-b px-5 py-4 font-heading !text-base !font-medium text-heading-foreground"
            style="font-weight: inherit;"
            variant="link"
            href="{{ route('dashboard.user.openai.documents.all') }}"
        >
            {{ __('Recently Launched') }}
            <x-tabler-chevron-right class="size-4" />
        </x-button>

        <!-- Recent launched -->
        <div
            class="recent-lunched-container flex flex-col"
            x-html="recentLunchedDocs"
        >
            <div class="block p-6 text-center font-medium text-heading-foreground">
                <h3 class="mb-2">{{ __('There is no recent lunch.') }}</h3>
            </div>
        </div>
    </div>
</div>
