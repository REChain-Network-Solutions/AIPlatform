@php
    use App\Helpers\Classes\MarketplaceHelper;

    $installed_extensions = [
        'ugc-factory' => [
            'label' => __('UGC Factory'),
            'description' => __('Realistic Actors for Social Media and Marketing'),
            'entryRoute' => MarketplaceHelper::isRegistered('ugc-factory') ? 'dashboard.user.ugc-factory.index' : null,
            'cardImage' => custom_theme_url('/assets/img/ugc-studio/ugc-factory-card.webp'),
            'status' => MarketplaceHelper::isRegistered('ugc-factory') ? 'installed' : 'not-installed',
        ],
        'ugc-creator' => [
            'label' => __('UGC Creator'),
            'description' => __('Create Custom UGC Content'),
            'entryRoute' => MarketplaceHelper::isRegistered('ugc-creator') ? 'dashboard.user.ugc-creator.index' : null,
            'cardImage' => custom_theme_url('/assets/img/ugc-studio/ugc-creator-card.webp'),
            'status' => MarketplaceHelper::isRegistered('ugc-creator') ? 'installed' : 'not-installed',
        ],
        'ugc-marketing' => [
            'label' => __('UGC Marketing'),
            'description' => __('Promote your products and apps'),
            'entryRoute' => MarketplaceHelper::isRegistered('ugc-creator') ? 'dashboard.user.ugc-creator.index' : null,
            'cardImage' => custom_theme_url('/assets/img/ugc-studio/ugc-marketing-card.webp'),
            'status' => MarketplaceHelper::isRegistered('ugc-creator') ? 'installed' : 'not-installed',
        ],
        'digital-twin' => [
            'label' => __('Digital Twin'),
            'description' => __('Create your own UGC Avatar'),
            'entryRoute' => MarketplaceHelper::isRegistered('ai-persona') ? 'dashboard.user.ai-persona.index' : null,
            'cardImage' => custom_theme_url('/assets/img/ugc-studio/digital-twin-card.webp'),
            'status' => MarketplaceHelper::isRegistered('ai-persona') ? 'installed' : 'not-installed',
        ],
    ];
@endphp

@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('UGC Studio'))
@section('titlebar_pretitle', '')
@section('titlebar_subtitle', __('Create UGC Content for social media and marketing.'))

@section('titlebar_actions')
    <x-button
        variant="ghost-shadow"
        href="#"
    >
        {{ __('UGC Videos') }}
    </x-button>

    @if (!$sources->isEmpty())
        <x-button :href="route($sources->first()->entryRoute)">
            <x-tabler-plus class="size-4" />
            {{ __('New UGC Content') }}
        </x-button>
    @endif
@endsection

@section('content')
    @php
        $serializeEntry = static function ($entry) {
            return [
                'source' => $entry['source'] ?? null,
                'source_label' => $entry['source_label'] ?? null,
                'id' => $entry['id'] ?? null,
                'title' => $entry['title'] ?? '—',
                'thumb_url' => $entry['thumb_url'] ?? null,
                'video_url' => $entry['video_url'] ?? null,
                'status' => $entry['status'] ?? null,
                'error' => $entry['error'] ?? null,
                'status_url' => $entry['status_url'] ?? null,
                'rename_url' => $entry['rename_url'] ?? null,
                'destroy_url' => $entry['destroy_url'] ?? null,
                'created_at' => isset($entry['created_at']) ? \Carbon\Carbon::parse($entry['created_at'])->diffForHumans() : null,
                'prompt' => $entry['prompt'] ?? null,
                'model' => $entry['model'] ?? null,
                'formatted_duration' => $entry['formatted_duration'] ?? null,
                'width' => $entry['width'] ?? null,
                'height' => $entry['height'] ?? null,
            ];
        };

        $todayEntries = $todayEntries ?? collect();
        $previousEntries = $previousEntries ?? collect();
        $paging = $paging ?? [];
        $perPage = $perPage ?? 12;

        $initialPayload = [
            'today' => $todayEntries->map($serializeEntry)->values()->all(),
            'previous' => $previousEntries->map($serializeEntry)->values()->all(),
            'paging' => $paging,
            'perPage' => $perPage,
            'loadMoreUrl' => route('dashboard.user.ugc-studio.outputs'),
        ];

        $hasAnyEntry = $todayEntries->isNotEmpty() || $previousEntries->isNotEmpty();
    @endphp

    <div class="py-10">
        <div x-data="ugcStudio">
            @if ($sources->isEmpty())
                <x-empty-state
                    icon="video"
                    title="{{ __('No UGC tools installed yet') }}"
                    description="{{ __('Install a UGC parent extension (e.g. UGC Factory) to start creating videos.') }}"
                />
            @else
                <div class="flex flex-col">
                    <div
                        class="lqd-ugc-studio-sources-grid grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4"
                        x-show="aiToolsShow"
                        x-collapse
                    >
                        <script>
                            (() => {
                                const el = document.querySelector('.lqd-ugc-studio-sources-grid');
                                const hide = localStorage.getItem('ugcStudio.aiToolsShow') === 'false';
                                if (hide) {
                                    el.style.display = 'none';
                                }
                            })();
                        </script>
                        @foreach ($installed_extensions as $key => $extension)
                            <x-card
                                class="transition hover:-translate-y-1 hover:shadow-xl hover:shadow-black/5 active:translate-y-0"
                                class:body="p-2.5"
                                size="sm"
                            >
                                @if (!empty($extension['cardImage']))
                                    <figure
                                        class="relative mb-1 aspect-[1/0.62] overflow-hidden rounded-[10px]"
                                        aria-hidden="true"
                                    >
                                        <img
                                            class="size-full object-cover object-bottom transition group-hover/card:scale-105"
                                            aria-hidden="true"
                                            src="{{ $extension['cardImage'] }}"
                                            alt="{{ $extension['label'] }}"
                                        >

                                        @if ($extension['status'] !== 'installed')
                                            <div
                                                class="absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center gap-2 rounded-full border-foreground/5 bg-background/85 px-2.5 py-[5px] text-[12px] font-medium shadow-xl shadow-black/10 backdrop-blur-lg">
                                                <span class="inline-flex size-1.5 rounded-full bg-foreground/80"></span>
                                                {{ Auth::user()->isAdmin() ? __('Install Now') : __('Not Available') }}
                                            </div>
                                        @endif
                                    </figure>
                                @endif

                                <h6 class="mb-0 text-3xs font-semibold opacity-50">
                                    {{ $extension['description'] }}
                                </h6>
                                <h4 class="m-0 text-2xs font-semibold">
                                    {{ $extension['label'] }}
                                </h4>

                                @if ($extension['entryRoute'] && $extension['status'] === 'installed')
                                    <a
                                        class="absolute inset-0 z-1"
                                        href="{{ route($extension['entryRoute']) }}"
                                    ></a>
                                @endif

                                @if ($extension['status'] === 'not-installed' && Auth::user()->isAdmin())
                                    <a
                                        class="absolute inset-0 z-1"
                                        href="{{ route('dashboard.admin.marketplace.index') }}"
                                    ></a>
                                @endif
                            </x-card>
                        @endforeach
                    </div>

                    <button
                        class="mt-6 flex w-full items-center gap-8 py-2"
                        type="button"
                        @click.prevent="aiToolsShow = !aiToolsShow"
                    >
                        <span class="inline-block h-px grow bg-current opacity-5"></span>
                        <span class="inline-flex items-center gap-1 text-2xs font-medium">
                            <span x-text="aiToolsShow ? '{{ __('Hide AI Tools') }}' : '{{ __('Show AI Tools') }}'">
                                {{ __('Hide AI Tools') }}
                            </span>
                            <x-tabler-chevron-up
                                class="size-4 transition"
                                ::class="{ 'rotate-180': !aiToolsShow }"
                            />
                        </span>
                        <span class="inline-block h-px grow bg-current opacity-5"></span>
                    </button>
                </div>

                <div class="pt-12">
                    <div x-data="ugcStudioRecent">
                        {{-- Empty state — visible when no entries on either SSR or
                             after a bulk delete. x-cloak only when SSR had entries,
                             so the no-content case renders immediately. --}}
                        <div
                            x-show="todayEntries.length === 0 && previousEntries.length === 0"
                            @if ($hasAnyEntry) x-cloak @endif
                        >
                            <x-empty-state
                                icon="tabler-video-off"
                                title="{{ __('No videos yet') }}"
                                description="{{ __('Generate your first UGC video to see it here.') }}"
                            />
                        </div>

                        @if ($hasAnyEntry)
                            {{-- Skeleton — only rendered when SSR actually has entries to hydrate. --}}
                            <div
                                class="flex flex-col gap-5"
                                x-show="!loaded"
                            >
                                @foreach (['today', 'previous'] as $skeletonBucket)
                                    <div>
                                        <div class="mb-5 flex items-center justify-between border-b py-2.5">
                                            <div class="lqd-loading-skeleton lqd-is-loading relative h-3 w-24 overflow-hidden rounded">
                                                <div
                                                    class="size-full"
                                                    data-lqd-skeleton-el
                                                ></div>
                                            </div>
                                            <div class="lqd-loading-skeleton lqd-is-loading relative h-3 w-16 overflow-hidden rounded">
                                                <div
                                                    class="size-full"
                                                    data-lqd-skeleton-el
                                                ></div>
                                            </div>
                                        </div>
                                        <div class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
                                            @foreach (range(1, 4) as $i)
                                                <x-card
                                                    class="overflow-hidden"
                                                    size="none"
                                                >
                                                    <div class="lqd-loading-skeleton lqd-is-loading relative aspect-[1/0.7] w-full overflow-hidden">
                                                        <div
                                                            class="size-full"
                                                            data-lqd-skeleton-el
                                                        ></div>
                                                    </div>
                                                    <div class="space-y-2 p-3">
                                                        <div class="lqd-loading-skeleton lqd-is-loading relative h-[1lh] w-3/4 overflow-hidden rounded text-2xs">
                                                            <div
                                                                class="size-full"
                                                                data-lqd-skeleton-el
                                                            ></div>
                                                        </div>
                                                        <div class="lqd-loading-skeleton lqd-is-loading relative h-[1lh] w-1/3 overflow-hidden rounded text-2xs">
                                                            <div
                                                                class="size-full"
                                                                data-lqd-skeleton-el
                                                            ></div>
                                                        </div>
                                                    </div>
                                                </x-card>
                                            @endforeach
                                        </div>
                                    </div>
                                @endforeach
                            </div>
                        @endif

                        <div
                            class="flex flex-col gap-5"
                            x-show="loaded"
                            x-cloak
                        >
                            {{-- Today --}}
                            <section x-show="todayEntries.length > 0">
                                <div class="mb-5 flex items-center justify-between border-b py-2.5">
                                    <p class="mb-0 text-[12px] font-semibold">
                                        {{ __('Today') }}
                                    </p>
                                    <x-button
                                        class="text-[12px] font-semibold text-primary hover:underline"
                                        variant="link"
                                        size="none"
                                        type="button"
                                        @click.prevent="toggleSelectAll('today')"
                                    >
                                        <span x-text="hasSelectedToday ? '{{ __('Deselect All') }}' : '{{ __('Select All') }}'"></span>
                                    </x-button>
                                </div>

                                <div class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
                                    <template
                                        x-for="entry in todayEntries"
                                        :key="entryKey(entry)"
                                    >
                                        @include('panel.user.ugc-studio.partials.entry-card')
                                    </template>
                                </div>

                                <div
                                    class="mt-4 flex items-center justify-center"
                                    x-ref="todaySentinel"
                                >
                                    <template x-if="loadingMore.today">
                                        <x-shimmer-text class="text-2xs">
                                            {{ __('Loading more…') }}
                                        </x-shimmer-text>
                                    </template>
                                    <template x-if="!loadingMore.today && _loadedMore.today && exhausted.today">
                                        <p class="m-0 text-2xs opacity-50">{{ __('All items loaded') }}</p>
                                    </template>
                                </div>
                            </section>

                            {{-- Created Previously --}}
                            <section x-show="previousEntries.length > 0">
                                <div class="mb-5 flex items-center justify-between border-b py-2.5">
                                    <p class="mb-0 text-[12px] font-semibold">
                                        {{ __('Created Previously') }}
                                    </p>
                                    <x-button
                                        class="text-[12px] font-semibold text-primary hover:underline"
                                        variant="link"
                                        size="none"
                                        type="button"
                                        @click.prevent="toggleSelectAll('previous')"
                                    >
                                        <span x-text="hasSelectedPrevious ? '{{ __('Deselect All') }}' : '{{ __('Select All') }}'"></span>
                                    </x-button>
                                </div>

                                <div class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
                                    <template
                                        x-for="entry in previousEntries"
                                        :key="entryKey(entry)"
                                    >
                                        @include('panel.user.ugc-studio.partials.entry-card')
                                    </template>
                                </div>

                                <div
                                    class="mt-4 flex items-center justify-center"
                                    x-ref="previousSentinel"
                                >
                                    <template x-if="loadingMore.previous">
                                        <x-shimmer-text class="text-2xs">
                                            {{ __('Loading more…') }}
                                        </x-shimmer-text>
                                    </template>
                                    <template x-if="!loadingMore.previous && _loadedMore.previous && exhausted.previous">
                                        <p class="m-0 text-2xs opacity-50">{{ __('All items loaded') }}</p>
                                    </template>
                                </div>
                            </section>
                        </div>

                        {{-- Bulk action bar --}}
                        <div
                            class="pointer-events-none fixed bottom-8 end-0 start-0 z-20 transition-all max-lg:bottom-[calc(var(--bottom-menu-height)+1rem)] lg:start-[--navbar-width]"
                            x-show="totalSelected > 0"
                            x-cloak
                            x-transition.scale-95
                        >
                            <div class="container">
                                <form
                                    class="pointer-events-auto flex flex-col items-center justify-between gap-1 rounded-full border border-foreground/5 bg-background px-6 py-4 shadow-xl shadow-black/5 md:flex-row md:py-1 md:pe-1"
                                    x-data="{ selectedAction: 'delete' }"
                                    @submit.prevent="bulkDelete"
                                >
                                    <span
                                        class="text-2xs font-medium"
                                        x-text="totalSelected + ' {{ __('selected') }}'"
                                    ></span>

                                    <div class="flex items-center gap-2">
                                        <x-forms.input
                                            class="w-full rounded-full pe-12 md:w-auto md:pe-12"
                                            type="select"
                                            size="md"
                                            x-model="selectedAction"
                                        >
                                            <option value="delete">
                                                {{ __('Move to Trash') }}
                                            </option>
                                        </x-forms.input>

                                        <x-button
                                            type="submit"
                                            ::disabled="deleting"
                                        >
                                            <span x-text="deleting ? '{{ __('Deleting...') }}' : '{{ __('Apply') }}'"></span>
                                        </x-button>

                                        <x-button
                                            variant="outline"
                                            hover-variant="none"
                                            type="button"
                                            @click="cancelSelection"
                                        >
                                            {{ __('Cancel') }}
                                        </x-button>
                                    </div>
                                </form>
                            </div>
                        </div>

                        @include('panel.user.ugc-studio.partials.video-detail-modal')

                        {{-- Rename modal --}}
                        <x-modal
                            class:modal-content="w-[min(100%,520px)]"
                            class:modal-body="p-6"
                            id="ugc-studio-rename-modal"
                            title="{{ __('Rename Video') }}"
                        >
                            <x-slot:modal>
                                <form
                                    class="flex flex-col gap-4"
                                    @submit.prevent="submitRename()"
                                >
                                    <x-forms.input
                                        type="text"
                                        label="{{ __('Title') }}"
                                        name="title"
                                        x-model="renameModal.title"
                                        maxlength="160"
                                        required
                                    />

                                    <p
                                        class="m-0 text-2xs text-red-500"
                                        x-show="renameModal.error"
                                        x-text="renameModal.error"
                                    ></p>

                                    <div class="flex justify-end gap-2 border-t pt-3">
                                        <x-button
                                            variant="outline"
                                            @click.prevent="closeRenameModal()"
                                        >
                                            {{ __('Cancel') }}
                                        </x-button>
                                        <x-button
                                            tag="button"
                                            type="submit"
                                            ::disabled="renameModal.submitting"
                                        >
                                            <span x-show="!renameModal.submitting">{{ __('Save') }}</span>
                                            <span x-show="renameModal.submitting">{{ __('Saving...') }}</span>
                                        </x-button>
                                    </div>
                                </form>
                            </x-slot:modal>
                        </x-modal>
                    </div>
                </div>
            @endif
        </div>
    </div>
@endsection

@push('script')
    <script src="{{ custom_theme_url('/assets/libs/fslightbox/fslightbox.js') }}"></script>
    <script>
        document.addEventListener('alpine:init', () => {
            Alpine.data('ugcStudio', () => ({
                aiToolsShow: Alpine.$persist(true).as('ugcStudio.aiToolsShow'),
            }));

            const POLL_INTERVAL = 4000;

            Alpine.data('ugcStudioRecent', (initial = @json($initialPayload)) => ({
                todayEntries: Array.isArray(initial.today) ? initial.today : [],
                previousEntries: Array.isArray(initial.previous) ? initial.previous : [],
                paging: initial.paging || {},
                perPage: initial.perPage || 12,
                loadMoreUrl: initial.loadMoreUrl || '',
                loaded: false,
                loadingMore: {
                    today: false,
                    previous: false
                },
                _loadedMore: {
                    today: false,
                    previous: false
                },
                _observers: {},
                _timer: null,
                _seenCompleted: new Set(),
                _refreshSeq: 0,
                selectedKeys: [],
                deleting: false,
                modalOpen: false,
                modalEntry: null,

                openVideoModal(entry) {
                    if (!entry || entry.status !== 'completed' || !entry.video_url) return;
                    this.modalEntry = entry;
                    this.modalOpen = true;
                    document.documentElement.classList.add('overflow-hidden');
                },

                closeVideoModal() {
                    this.modalOpen = false;
                    document.documentElement.classList.remove('overflow-hidden');
                    this.$nextTick(() => {
                        const videoEl = this.$refs.modalVideo;
                        if (videoEl) {
                            videoEl.pause();
                            videoEl.removeAttribute('src');
                        }
                        this.modalEntry = null;
                    });
                },

                renameModal: {
                    entryKey: null,
                    title: '',
                    submitting: false,
                    error: '',
                },

                init() {
                    // Mark already-completed entries so we don't fire refreshFsLightbox for them.
                    this.allEntries().forEach((e) => {
                        if (e.status === 'completed') {
                            this._seenCompleted.add(this.entryKey(e));
                        }
                    });

                    this.$nextTick(() => {
                        this.loaded = true;
                        this.observeBucket('today');
                        this.observeBucket('previous');
                    });

                    this.scheduleTick();
                },

                // ---------- helpers ----------
                entryKey(entry) {
                    return `${entry.source}-${entry.id}`;
                },

                allEntries() {
                    return [...this.todayEntries, ...this.previousEntries];
                },

                downloadFilename(entry) {
                    const title = (entry.title || 'ugc-video').toString().trim();
                    const slug = title
                        .toLowerCase()
                        .replace(/[^a-z0-9]+/g, '-')
                        .replace(/^-+|-+$/g, '')
                        .slice(0, 80) || 'ugc-video';

                    let ext = 'mp4';
                    try {
                        const path = new URL(entry.video_url || '', window.location.origin).pathname;
                        const match = path.match(/\.([a-z0-9]{2,5})$/i);
                        if (match) ext = match[1];
                    } catch (_) {
                        /* fall through */
                    }

                    return `${slug}.${ext}`;
                },

                bucketEntries(bucket) {
                    return bucket === 'today' ? this.todayEntries : this.previousEntries;
                },

                replaceBucket(bucket, next) {
                    if (bucket === 'today') {
                        this.todayEntries = next;
                    } else {
                        this.previousEntries = next;
                    }
                },

                // ---------- load-more ----------
                get exhausted() {
                    const isExhausted = (bucket) => {
                        for (const key in this.paging) {
                            if (this.paging[key]?.[bucket]?.has_more) return false;
                        }
                        return true;
                    };
                    return {
                        today: isExhausted('today'),
                        previous: isExhausted('previous'),
                    };
                },

                observeBucket(bucket) {
                    const ref = bucket === 'today' ? this.$refs.todaySentinel : this.$refs.previousSentinel;
                    if (!ref || this.exhausted[bucket]) return;

                    if (typeof IntersectionObserver === 'undefined') return;

                    const observer = new IntersectionObserver((entries) => {
                        if (entries.some((e) => e.isIntersecting)) {
                            this.loadMore(bucket);
                        }
                    }, {
                        rootMargin: '200px 0px'
                    });

                    observer.observe(ref);
                    this._observers[bucket] = observer;
                },

                async loadMore(bucket) {
                    if (this.loadingMore[bucket] || this.exhausted[bucket]) return;

                    this.loadingMore[bucket] = true;
                    try {
                        for (const sourceKey of Object.keys(this.paging)) {
                            const state = this.paging[sourceKey]?.[bucket];
                            if (!state?.has_more) continue;

                            const url =
                                `${this.loadMoreUrl}?source=${encodeURIComponent(sourceKey)}&bucket=${encodeURIComponent(bucket)}&page=${state.next_page}`;
                            const res = await fetch(url, {
                                headers: {
                                    Accept: 'application/json'
                                }
                            });
                            if (!res.ok) continue;
                            const json = await res.json();
                            if (!Array.isArray(json.entries)) continue;

                            const next = this.bucketEntries(bucket).concat(this.normalizeEntries(json.entries));
                            this.replaceBucket(bucket, next);

                            this.paging[sourceKey][bucket] = {
                                has_more: !!json.has_more,
                                next_page: json.next_page,
                            };
                        }
                        this._loadedMore[bucket] = true;
                    } finally {
                        this.loadingMore[bucket] = false;
                    }

                    if (this.exhausted[bucket] && this._observers[bucket]) {
                        this._observers[bucket].disconnect();
                        delete this._observers[bucket];
                    }
                },

                normalizeEntries(raw) {
                    return raw.map((e) => ({
                        source: e.source ?? null,
                        source_label: e.source_label ?? null,
                        id: e.id ?? null,
                        title: e.title ?? '—',
                        thumb_url: e.thumb_url ?? null,
                        video_url: e.video_url ?? null,
                        status: e.status ?? null,
                        error: e.error ?? null,
                        status_url: e.status_url ?? null,
                        rename_url: e.rename_url ?? null,
                        destroy_url: e.destroy_url ?? null,
                        created_at: e.created_at ?? null,
                        prompt: e.prompt ?? null,
                        model: e.model ?? null,
                        formatted_duration: e.formatted_duration ?? null,
                        width: e.width ?? null,
                        height: e.height ?? null,
                    }));
                },

                // ---------- selection ----------
                get totalSelected() {
                    return this.selectedKeys.length;
                },

                isSelected(entry) {
                    return this.selectedKeys.includes(this.entryKey(entry));
                },

                toggleEntrySelection(entry) {
                    const key = this.entryKey(entry);
                    if (this.selectedKeys.includes(key)) {
                        this.selectedKeys = this.selectedKeys.filter((k) => k !== key);
                    } else {
                        this.selectedKeys.push(key);
                    }
                },

                get hasSelectedToday() {
                    if (this.todayEntries.length === 0) return false;
                    return this.todayEntries.every((e) => this.selectedKeys.includes(this.entryKey(e)));
                },

                get hasSelectedPrevious() {
                    if (this.previousEntries.length === 0) return false;
                    return this.previousEntries.every((e) => this.selectedKeys.includes(this.entryKey(e)));
                },

                toggleSelectAll(bucket) {
                    const entries = this.bucketEntries(bucket);
                    const keys = entries.map((e) => this.entryKey(e));
                    const all = bucket === 'today' ? this.hasSelectedToday : this.hasSelectedPrevious;

                    if (all) {
                        this.selectedKeys = this.selectedKeys.filter((k) => !keys.includes(k));
                    } else {
                        const merged = new Set(this.selectedKeys);
                        keys.forEach((k) => merged.add(k));
                        this.selectedKeys = Array.from(merged);
                    }
                },

                cancelSelection() {
                    this.selectedKeys = [];
                },

                async bulkDelete() {
                    if (this.deleting || this.selectedKeys.length === 0) return;

                    const message = this.selectedKeys.length === 1 ?
                        '{{ __('Delete the selected video? This action can\'t be undone.') }}' :
                        '{{ __('Delete :count videos? This action can\'t be undone.') }}'.replace(':count', String(this.selectedKeys.length));

                    if (!window.confirm(message)) return;

                    this.deleting = true;
                    const csrf = document.querySelector('meta[name="csrf-token"]')?.content || '';
                    const targets = this.allEntries().filter((e) => this.selectedKeys.includes(this.entryKey(e)));

                    try {
                        await Promise.all(targets.map((entry) => {
                            if (!entry.destroy_url) return Promise.resolve();
                            return fetch(entry.destroy_url, {
                                method: 'DELETE',
                                headers: {
                                    Accept: 'application/json',
                                    'X-CSRF-TOKEN': csrf
                                },
                            }).catch(() => null);
                        }));

                        const keysToDrop = new Set(this.selectedKeys);
                        this.todayEntries = this.todayEntries.filter((e) => !keysToDrop.has(this.entryKey(e)));
                        this.previousEntries = this.previousEntries.filter((e) => !keysToDrop.has(this.entryKey(e)));
                        this.selectedKeys = [];
                    } finally {
                        this.deleting = false;
                    }
                },

                // ---------- single rename ----------
                openRenameModal(entry) {
                    this.renameModal.entryKey = this.entryKey(entry);
                    this.renameModal.title = entry.title || '';
                    this.renameModal.error = '';
                    this.renameModal.submitting = false;
                    this._toggleRenameModal(true);
                },

                closeRenameModal() {
                    this._toggleRenameModal(false);
                    this.renameModal.entryKey = null;
                    this.renameModal.error = '';
                },

                _toggleRenameModal(state) {
                    const el = document.querySelector('#ugc-studio-rename-modal');
                    if (!el) return;
                    const data = Alpine.$data(el);
                    if (data) data.modalOpen = state;
                },

                async submitRename() {
                    const title = (this.renameModal.title || '').trim();
                    if (!title) {
                        this.renameModal.error = '{{ __('Title is required.') }}';
                        return;
                    }

                    const entry = this.allEntries().find((e) => this.entryKey(e) === this.renameModal.entryKey);
                    if (!entry || !entry.rename_url) {
                        this.renameModal.error = '{{ __('This video can\'t be renamed.') }}';
                        return;
                    }

                    this.renameModal.submitting = true;
                    this.renameModal.error = '';
                    const csrf = document.querySelector('meta[name="csrf-token"]')?.content || '';

                    try {
                        const res = await fetch(entry.rename_url, {
                            method: 'PATCH',
                            headers: {
                                Accept: 'application/json',
                                'Content-Type': 'application/json',
                                'X-CSRF-TOKEN': csrf,
                            },
                            body: JSON.stringify({
                                title
                            }),
                        });

                        const json = await res.json().catch(() => ({}));

                        if (!res.ok) {
                            this.renameModal.error = json.error || json.message || '{{ __('Could not rename the video.') }}';
                            return;
                        }

                        const apply = (list) => list.map((e) => this.entryKey(e) === this.renameModal.entryKey ? {
                                ...e,
                                title: json?.data?.title ?? title
                            } :
                            e);

                        this.todayEntries = apply(this.todayEntries);
                        this.previousEntries = apply(this.previousEntries);

                        this.closeRenameModal();
                    } catch (err) {
                        this.renameModal.error = err?.message || '{{ __('Network error.') }}';
                    } finally {
                        this.renameModal.submitting = false;
                    }
                },

                // ---------- single delete (per-card dropdown) ----------
                async deleteEntry(entry) {
                    if (!entry || !entry.destroy_url) return;

                    const message = entry.title ?
                        '{{ __('Delete ":title"? This action can\'t be undone.') }}'.replace(':title', entry.title) :
                        '{{ __('Delete this video? This action can\'t be undone.') }}';

                    if (!window.confirm(message)) return;

                    const csrf = document.querySelector('meta[name="csrf-token"]')?.content || '';
                    const key = this.entryKey(entry);

                    try {
                        const res = await fetch(entry.destroy_url, {
                            method: 'DELETE',
                            headers: {
                                Accept: 'application/json',
                                'X-CSRF-TOKEN': csrf
                            },
                        });

                        if (!res.ok) {
                            const json = await res.json().catch(() => ({}));
                            const msg = json.error || json.message || '{{ __('Could not delete the video.') }}';
                            if (typeof window.toastr !== 'undefined') {
                                window.toastr.error(msg);
                            } else {
                                window.alert(msg);
                            }
                            return;
                        }

                        this.todayEntries = this.todayEntries.filter((e) => this.entryKey(e) !== key);
                        this.previousEntries = this.previousEntries.filter((e) => this.entryKey(e) !== key);
                        this.selectedKeys = this.selectedKeys.filter((k) => k !== key);
                    } catch (err) {
                        const msg = err?.message || '{{ __('Network error.') }}';
                        if (typeof window.toastr !== 'undefined') {
                            window.toastr.error(msg);
                        } else {
                            window.alert(msg);
                        }
                    }
                },

                // ---------- polling for in-flight generations ----------
                scheduleTick() {
                    if (this._timer) clearTimeout(this._timer);
                    if (!this.hasPending()) return;
                    this._timer = setTimeout(() => this.pollPending(), POLL_INTERVAL);
                },

                hasPending() {
                    return this.allEntries().some((e) => e.status !== 'completed' && e.status !== 'failed');
                },

                async pollPending() {
                    const pending = this.allEntries().filter((e) => e.status !== 'completed' && e.status !== 'failed');
                    await Promise.all(pending.map((e) => this.refreshEntry(e)));
                    this.scheduleTick();
                },

                async refreshEntry(entry) {
                    if (!entry.status_url) return;

                    try {
                        const res = await fetch(entry.status_url, {
                            headers: {
                                Accept: 'application/json'
                            }
                        });
                        if (!res.ok) return;
                        const json = await res.json();
                        const data = json?.data;
                        if (!data) return;

                        const key = this.entryKey(entry);
                        const patch = (list) => list.map((e) => this.entryKey(e) === key ? {
                                ...e,
                                status: data.status ?? e.status,
                                error: data.error ?? e.error,
                                video_url: data.video_url ?? e.video_url,
                                thumb_url: data.thumb_url ?? e.thumb_url,
                                title: data.title ?? e.title,
                                formatted_duration: data.formatted_duration ?? e.formatted_duration,
                                prompt: data.prompt ?? e.prompt,
                                model: data.model ?? e.model,
                            } :
                            e);

                        this.todayEntries = patch(this.todayEntries);
                        this.previousEntries = patch(this.previousEntries);

                        if (data.status === 'completed' && !this._seenCompleted.has(key)) {
                            this._seenCompleted.add(key);
                            if (typeof window.refreshFsLightbox === 'function') {
                                window.refreshFsLightbox();
                            }
                        }
                    } catch (_) {
                        // soft fail; next tick will retry
                    }
                },
            }));
        });
    </script>
@endpush
