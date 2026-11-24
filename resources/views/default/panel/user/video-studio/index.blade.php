@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('Video Studio'))
@section('titlebar_pretitle', '')
@section('titlebar_subtitle', __('All your AI video tools and generated outputs in one place.'))

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
                'created_at' => $entry['created_at'] ?? null,
                'prompt' => $entry['prompt'] ?? null,
                'model' => $entry['model'] ?? null,
                'resolution' => $entry['resolution'] ?? null,
                'aspect_ratio' => $entry['aspect_ratio'] ?? null,
                'formatted_duration' => $entry['formatted_duration'] ?? null,
                'template_name' => $entry['template_name'] ?? null,
                'project_name' => $entry['project_name'] ?? null,
                'source_language' => $entry['source_language'] ?? null,
                'target_language' => $entry['target_language'] ?? null,
                'width' => $entry['width'] ?? null,
                'height' => $entry['height'] ?? null,
                '_ts' => $entry['_ts'] ?? 0,
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
            'loadMoreUrl' => route('dashboard.user.video-studio.outputs'),
        ];

        $hasAnyEntry = $todayEntries->isNotEmpty() || $previousEntries->isNotEmpty();
    @endphp

    <div class="py-10">
        <div x-data="videoStudio">
            @if ($tools->isEmpty())
                <x-empty-state
                    icon="video"
                    title="{{ __('No video tools installed yet') }}"
                    description="{{ __('Install an AI video extension (Video Pro, Video Editor, Video Dubbing, AI Captions) to start creating videos.') }}"
                />
            @else
                <div class="flex flex-col">
                    <div
                        class="lqd-video-studio-tools-grid grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4"
                        x-show="aiToolsShow"
                        x-collapse
                    >
                        <script>
                            (() => {
                                const el = document.querySelector('.lqd-video-studio-tools-grid');
                                const hide = localStorage.getItem('videoStudio.aiToolsShow') === 'false';
                                if (hide && el) {
                                    el.style.display = 'none';
                                }
                            })();
                        </script>
                        @foreach ($tools as $tool)
                            <x-card
                                class="transition hover:-translate-y-1 hover:shadow-xl hover:shadow-black/5 active:translate-y-0"
                                class:body="p-2.5"
                                size="sm"
                            >
                                @if (!empty($tool['card_image']))
                                    <figure
                                        class="mb-1 aspect-[1/0.62] overflow-hidden rounded-[10px]"
                                        aria-hidden="true"
                                    >
                                        <img
                                            class="size-full object-cover object-bottom transition group-hover/card:scale-105"
                                            aria-hidden="true"
                                            src="{{ $tool['card_image'] }}"
                                            alt="{{ $tool['label'] }}"
                                        >
                                    </figure>
                                @else
                                    <figure
                                        class="mb-1 inline-grid aspect-[1/0.62] place-items-center overflow-hidden rounded-[10px] bg-foreground/5"
                                        aria-hidden="true"
                                    >
                                        <x-tabler-video class="size-10 opacity-30" />
                                    </figure>
                                @endif
                                <h6 class="mb-0 text-3xs font-semibold opacity-50">
                                    {{ $tool['description'] }}
                                </h6>
                                <h4 class="m-0 text-2xs font-semibold">
                                    {{ $tool['label'] }}
                                </h4>
                                <a
                                    class="absolute inset-0 z-1"
                                    href="{{ route($tool['entry_route']) }}"
                                ></a>
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
                    <div x-data="videoStudioRecent">
                        <div
                            x-show="todayEntries.length === 0 && previousEntries.length === 0"
                            @if ($hasAnyEntry) x-cloak @endif
                        >
                            <x-empty-state
                                icon="tabler-video-off"
                                title="{{ __('No videos yet') }}"
                                description="{{ __('Generate your first video to see it here.') }}"
                            />
                        </div>

                        @if ($hasAnyEntry)
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
                                        @include('panel.user.video-studio.partials.entry-card')
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

                            <section x-show="previousEntries.length > 0">
                                <div class="mb-5 flex items-center justify-between border-b py-2.5">
                                    <p class="mb-0 text-[12px] font-semibold">
                                        {{ __('Recent Videos') }}
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
                                        @include('panel.user.video-studio.partials.entry-card')
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

                        @include('panel.user.video-studio.partials.video-detail-modal')

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
                    </div>
                </div>
            @endif
        </div>
    </div>
@endsection

@push('script')
    <script>
        document.addEventListener('alpine:init', () => {
            Alpine.data('videoStudio', () => ({
                aiToolsShow: Alpine.$persist(true).as('videoStudio.aiToolsShow'),
            }));

            const POLL_INTERVAL = 4000;

            Alpine.data('videoStudioRecent', (initial = @json($initialPayload)) => ({
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

                init() {
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

                entryKey(entry) {
                    return `${entry.source}-${entry.id}`;
                },

                allEntries() {
                    return [...this.todayEntries, ...this.previousEntries];
                },

                downloadFilename(entry) {
                    const title = (entry.title || 'video').toString().trim();
                    const slug = title
                        .toLowerCase()
                        .replace(/[^a-z0-9]+/g, '-')
                        .replace(/^-+|-+$/g, '')
                        .slice(0, 80) || 'video';

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

                            const next = this.sortByDate(this.bucketEntries(bucket).concat(this.normalizeEntries(json.entries)));
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
                        resolution: e.resolution ?? null,
                        aspect_ratio: e.aspect_ratio ?? null,
                        formatted_duration: e.formatted_duration ?? null,
                        template_name: e.template_name ?? null,
                        project_name: e.project_name ?? null,
                        source_language: e.source_language ?? null,
                        target_language: e.target_language ?? null,
                        width: e.width ?? null,
                        height: e.height ?? null,
                        _ts: e._ts ?? 0,
                    }));
                },

                sortByDate(list) {
                    return [...list].sort((a, b) => (b._ts ?? 0) - (a._ts ?? 0));
                },

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

                async openRenameModal(entry) {
                    if (!entry || !entry.rename_url) return;

                    const current = entry.title || '';
                    const next = window.prompt('{{ __('Rename video') }}', current);
                    if (next === null) return;

                    const trimmed = next.trim();
                    if (trimmed === '' || trimmed === current) return;

                    const csrf = document.querySelector('meta[name="csrf-token"]')?.content || '';
                    const key = this.entryKey(entry);
                    const body = new FormData();
                    body.append('name', trimmed);

                    try {
                        const res = await fetch(entry.rename_url, {
                            method: 'POST',
                            headers: {
                                Accept: 'application/json',
                                'X-CSRF-TOKEN': csrf,
                            },
                            body,
                        });

                        if (!res.ok) {
                            const msg = '{{ __('Could not rename the video.') }}';
                            if (typeof window.toastr !== 'undefined') {
                                window.toastr.error(msg);
                            } else {
                                window.alert(msg);
                            }
                            return;
                        }

                        const json = await res.json().catch(() => ({}));
                        const newTitle = json.title || trimmed;
                        const patch = (list) => list.map((e) => this.entryKey(e) === key ? { ...e, title: newTitle } : e);
                        this.todayEntries = patch(this.todayEntries);
                        this.previousEntries = patch(this.previousEntries);
                        if (this.modalEntry && this.entryKey(this.modalEntry) === key) {
                            this.modalEntry = { ...this.modalEntry, title: newTitle };
                        }
                    } catch (err) {
                        const msg = err?.message || '{{ __('Network error.') }}';
                        if (typeof window.toastr !== 'undefined') {
                            window.toastr.error(msg);
                        } else {
                            window.alert(msg);
                        }
                    }
                },

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
                            headers: {
                                Accept: 'application/json',
                                'X-CSRF-TOKEN': csrf
                            },
                        });

                        if (!res.ok) {
                            const msg = '{{ __('Could not delete the video.') }}';
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
                        const data = json?.data ?? json;
                        if (!data) return;

                        const key = this.entryKey(entry);
                        const patch = (list) => list.map((e) => this.entryKey(e) === key ? {
                                ...e,
                                status: data.status ?? e.status,
                                error: data.error ?? e.error,
                                video_url: data.video_url ?? data.output_url ?? e.video_url,
                                thumb_url: data.thumb_url ?? data.thumbnail_url ?? e.thumb_url,
                                title: data.title ?? e.title,
                            } :
                            e);

                        this.todayEntries = patch(this.todayEntries);
                        this.previousEntries = patch(this.previousEntries);

                        if (data.status === 'completed' && !this._seenCompleted.has(key)) {
                            this._seenCompleted.add(key);
                        }
                    } catch (_) {
                        /* soft fail */
                    }
                },
            }));
        });
    </script>
@endpush
