@php
    if ($style === 'compact') {
        $container_class = [
            'header-search-results-container max-h-[min(440px,80vh)] overflow-y-auto rounded-xl bg-background shadow-xl shadow-black/5 absolute top-full inset-x-0 z-50 mt-4',
        ];
    } else {
        $container_class = ['header-search-results-container h-full overflow-y-auto'];
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
    {{-- Search results panel --}}
    <div
        class="header-search-results h-full w-full"
        x-cloak
        x-show="doneSearching"
    >
        {{-- AI mode indicator and response --}}
        <template x-if="aiMode">
            <div>
                <div class="border-b px-5 py-4">
                    <div class="flex items-center justify-between gap-4">
                        <div class="flex items-center gap-3">
                            <div class="flex size-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/15 text-violet-600 dark:text-violet-400">
                                <svg
                                    class="size-4"
                                    xmlns="http://www.w3.org/2000/svg"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    stroke-width="1.75"
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                >
                                    <path
                                        stroke="none"
                                        d="M0 0h24v24H0z"
                                        fill="none"
                                    />
                                    <path d="M9.5 3h5m-2.5 0v4m-5 1l-3.5 -3.5m1.5 3.5l2.5 2.5m8 -6l3.5 3.5m-3.5 0l-2.5 2.5m-1 5.5v5m-2 -2l2 2 2 -2" />
                                    <circle
                                        cx="12"
                                        cy="12"
                                        r="3"
                                    />
                                </svg>
                            </div>
                            <div class="text-start">
                                <p class="mb-1 text-sm font-semibold text-heading-foreground">{{ __('AI Assistant') }}</p>
                                <p class="m-0 text-xs text-foreground/50">{{ __('Ask anything — type your question') }}</p>
                            </div>
                        </div>
                        <kbd class="shrink-0 rounded-lg border bg-background px-2.5 py-0.5 font-mono text-2xs text-foreground/50">
                            ↵ {{ __('Enter') }}
                        </kbd>
                    </div>
                </div>

                <div
                    class="min-h-[80px] px-5 py-4 text-start"
                    x-show="aiResponse || aiStreaming"
                >
                    <p
                        class="mb-1 whitespace-pre-wrap text-sm leading-relaxed text-foreground"
                        x-text="aiResponse"
                    ></p>
                    <span
                        class="m-0 inline-block h-4 w-0.5 animate-pulse bg-violet-500 align-middle"
                        x-show="aiStreaming"
                    ></span>
                </div>

                <div
                    class="px-5 py-8 text-center text-sm text-foreground/50"
                    x-show="!aiResponse && !aiStreaming"
                >
                    {{ __('Type a question and press Enter') }}
                </div>
            </div>
        </template>

        {{-- Grouped search results --}}
        <template x-if="!aiMode">
            <div class="h-full">
                <template x-if="filteredResults.length > 0">
                    <div>
                        <template
                            x-for="(group, groupIndex) in filteredResults"
                            :key="group.type"
                        >
                            <div class="border-b last:border-b-0">
                                <h4
                                    class="px-5 pb-1 pt-3 text-3xs font-semibold uppercase tracking-wider text-foreground/40"
                                    x-text="group.label"
                                ></h4>

                                <template
                                    x-for="(item, itemIndex) in group.items"
                                    :key="item.url + item.label"
                                >
                                    <a
                                        class="flex items-center gap-3 px-4 py-2.5 text-sm transition-colors hover:bg-heading-foreground/5 focus:outline-none"
                                        :href="item.url"
                                        :class="{
                                            'border-s-2 border-sky-600 bg-heading-foreground/5': flatItems.indexOf(item) === focusedIndex
                                        }"
                                        @click="trackActivity(item)"
                                    >
                                        <span
                                            class="flex size-8 shrink-0 items-center justify-center rounded-lg border bg-background text-foreground/60 [&>svg]:size-4"
                                            x-html="item.svg"
                                        ></span>
                                        <span class="min-w-0 flex-1 text-start">
                                            <span
                                                class="mb-1 block truncate font-medium text-heading-foreground"
                                                x-text="item.label"
                                            ></span>
                                            <span
                                                class="block truncate text-2xs text-foreground/50"
                                                x-text="item.context"
                                            ></span>
                                        </span>
                                        <svg
                                            class="size-3.5 shrink-0 text-foreground/30 rtl:-scale-x-100"
                                            xmlns="http://www.w3.org/2000/svg"
                                            viewBox="0 0 24 24"
                                            fill="none"
                                            stroke="currentColor"
                                            stroke-width="2"
                                            stroke-linecap="round"
                                            stroke-linejoin="round"
                                        >
                                            <path
                                                stroke="none"
                                                d="M0 0h24v24H0z"
                                                fill="none"
                                            />
                                            <path d="M9 6l6 6l-6 6" />
                                        </svg>
                                    </a>
                                </template>
                            </div>
                        </template>
                    </div>
                </template>

                <template x-if="filteredResults.length === 0">
                    <div class="flex h-full flex-col items-center justify-center gap-2 px-5 py-8 text-center">
                        <p class="text-sm font-medium">{{ __('No results') }}</p>
                        <p class="text-xs">{{ __('We have not found anything matching your search criteria.') }}</p>
                    </div>
                </template>
            </div>
        </template>
    </div>

    {{-- Last Activities / pending state --}}
    <div
        class="header-search-recents"
        x-cloak
        x-show="isSearching || pending"
    >
        <template x-if="lastActivities.length > 0">
            <div>
                <h3 class="m-0 w-full border-b px-5 py-3 text-start text-3xs font-semibold uppercase tracking-wider text-foreground/40">
                    {{ __('Last Activities') }}
                </h3>

                <template
                    x-for="activity in lastActivities"
                    :key="activity.url"
                >
                    <div class="group flex items-center gap-3 px-4 py-2.5 text-sm transition-colors hover:bg-heading-foreground/5">
                        <a
                            class="flex min-w-0 flex-1 items-center gap-3"
                            :href="activity.url"
                            @click="trackActivity(activity)"
                        >
                            <span
                                class="flex size-8 shrink-0 items-center justify-center rounded-lg border bg-background text-foreground/60 [&>svg]:size-4"
                                x-html="activity.svg"
                            ></span>
                            <span class="min-w-0 flex-1 text-start">
                                <span
                                    class="mb-1 block truncate font-medium text-heading-foreground"
                                    x-text="activity.label"
                                ></span>
                                <span
                                    class="block truncate text-2xs text-foreground/50"
                                    x-text="activity.context"
                                ></span>
                            </span>
                        </a>
                        <button
                            class="inline-grid size-[18px] shrink-0 cursor-pointer place-items-center rounded-full opacity-0 transition hover:scale-110 hover:bg-background hover:text-foreground group-hover:opacity-100"
                            type="button"
                            @click.prevent="deleteActivity(activity.url)"
                        >
                            <x-tabler-x class="size-3.5 opacity-60" />
                        </button>
                    </div>
                </template>
            </div>
        </template>

        <template x-if="lastActivities.length === 0">
            <div class="py-5">
                <p class="mb-3 px-4 text-3xs font-semibold uppercase tracking-wider text-foreground/40">{{ __('Quick Links') }}</p>
                <div>
                    @php
                        $quickLinks = [
                            ['label' => __('Dashboard'), 'context' => __('Overview & stats'), 'icon' => 'tabler-layout-dashboard', 'route' => 'dashboard.user.index'],
                            ['label' => __('Templates'), 'context' => __('Browse AI templates'), 'icon' => 'tabler-layout-2', 'route' => 'dashboard.user.generator.index'],
                            ['label' => __('Chat'), 'context' => __('Start a conversation'), 'icon' => 'tabler-message', 'route' => 'dashboard.user.openai.chat.chat'],
                            ['label' => __('Documents'), 'context' => __('Your generated content'), 'icon' => 'tabler-file-text', 'route' => 'dashboard.user.openai.documents.all'],
                        ];
                    @endphp
                    @foreach ($quickLinks as $link)
                        <a
                            class="flex items-center gap-3 px-4 py-2.5 text-sm transition-colors hover:bg-heading-foreground/5 focus:outline-none"
                            href="{{ route($link['route']) }}"
                        >
                            <span class="flex size-8 shrink-0 items-center justify-center rounded-lg border bg-background text-foreground/60">
                                <x-dynamic-component
                                    class="size-4"
                                    :component="$link['icon']"
                                    stroke-width="1.5"
                                />
                            </span>
                            <span class="min-w-0 flex-1 text-start">
                                <span class="block truncate font-medium text-heading-foreground">{{ $link['label'] }}</span>
                                <span class="block truncate text-2xs text-foreground/50">{{ $link['context'] }}</span>
                            </span>
                            <svg
                                class="size-3.5 shrink-0 text-foreground/30 rtl:-scale-x-100"
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                stroke-width="2"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                            >
                                <path
                                    stroke="none"
                                    d="M0 0h24v24H0z"
                                    fill="none"
                                />
                                <path d="M9 6l6 6l-6 6" />
                            </svg>
                        </a>
                    @endforeach
                </div>
            </div>
        </template>
    </div>
</div>
