@php
    $uid = $uid ?? uniqid();
    $modalId = 'elevenlabs-library-modal-' . $uid;
    $triggerLabel = $triggerLabel ?? __('Browse Voice Library');
    $triggerClass = @twMerge($triggerClass ?? '');
    $indexRoute = $indexRoute ?? 'dashboard.admin.settings.tts.elevenlabs.library.index';
    $addRoute = $addRoute ?? 'dashboard.admin.settings.tts.elevenlabs.library.add';
    $removeRoute = $removeRoute ?? 'dashboard.admin.settings.tts.elevenlabs.library.remove';
    $addedIds = \App\Models\Voice\ElevenlabVoice::query()->whereNull('user_id')->whereNotNull('voice_id')->pluck('voice_id')->values()->toArray();
@endphp

<div
    class="contents"
    x-data="elevenlabsSharedLibrary"
>
    <x-button
        @class($triggerClass)
        type="button"
        variant="outline"
        @click.prevent="toggleModal(true)"
    >
        <x-tabler-library class="size-4" />
        {{ $triggerLabel }}
    </x-button>

    <x-modal
        class:modal-body="p-0"
        class:modal-content="w-[min(100%,940px)] h-[min(690px,calc(100vh-4rem))] p-5 lg:px-8 lg:py-7"
        class:modal-title="text-[12px] font-semibold"
        class:close-btn="size-6 [&_svg]:size-4"
        class:modal-head="px-0 pt-1"
        id="{{ $modalId }}"
        disable-focus
        title="{{ __('Select Voice') }}"
    >
        <x-slot:modal>
            <div class="py-5">
                <x-forms.input
                    class="h-10 border-none bg-foreground/5"
                    class:container="col-span-4 mb-4"
                    type="search"
                    placeholder="{{ __('Search voices') }}"
                    x-model.debounce.400ms="filters.search"
                    @input.debounce.400ms="reload()"
                />

                <div class="mb-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    <x-forms.input
                        class="capitalize"
                        type="select"
                        x-model="filters.gender"
                        @change="reload()"
                    >
                        <option value="">{{ __('Any gender') }}</option>
                        <option value="male">{{ __('Male') }}</option>
                        <option value="female">{{ __('Female') }}</option>
                        <option value="neutral">{{ __('Neutral') }}</option>
                    </x-forms.input>

                    <x-forms.input
                        class="capitalize"
                        type="select"
                        x-model="filters.age"
                        @change="reload()"
                    >
                        <option value="">{{ __('Any age') }}</option>
                        <option value="young">{{ __('Young') }}</option>
                        <option value="middle_aged">{{ __('Middle aged') }}</option>
                        <option value="old">{{ __('Old') }}</option>
                    </x-forms.input>

                    <x-forms.input
                        class="capitalize"
                        type="select"
                        x-model="filters.language"
                        @change="reload()"
                    >
                        <option value="">{{ __('Any language') }}</option>
                        <option value="af">Afrikaans</option>
                        <option value="sq">Shqip</option>
                        <option value="am">አማርኛ</option>
                        <option value="ar">العربية</option>
                        <option value="hy">Հայերեն</option>
                        <option value="as">অসমীয়া</option>
                        <option value="az">Azərbaycan</option>
                        <option value="be">Беларуская</option>
                        <option value="bn">বাংলা</option>
                        <option value="bs">Bosanski</option>
                        <option value="bg">Български</option>
                        <option value="my">မြန်မာ</option>
                        <option value="ca">Català</option>
                        <option value="ceb">Cebuano</option>
                        <option value="zh">中文</option>
                        <option value="hr">Hrvatski</option>
                        <option value="cs">Čeština</option>
                        <option value="da">Dansk</option>
                        <option value="nl">Nederlands</option>
                        <option value="en">English</option>
                        <option value="et">Eesti</option>
                        <option value="fil">Filipino</option>
                        <option value="fi">Suomi</option>
                        <option value="fr">Français</option>
                        <option value="gl">Galego</option>
                        <option value="ka">ქართული</option>
                        <option value="de">Deutsch</option>
                        <option value="el">Ελληνικά</option>
                        <option value="gu">ગુજરાતી</option>
                        <option value="ht">Kreyòl ayisyen</option>
                        <option value="ha">Hausa</option>
                        <option value="he">עברית</option>
                        <option value="hi">हिन्दी</option>
                        <option value="hu">Magyar</option>
                        <option value="is">Íslenska</option>
                        <option value="ig">Igbo</option>
                        <option value="id">Bahasa Indonesia</option>
                        <option value="ga">Gaeilge</option>
                        <option value="it">Italiano</option>
                        <option value="ja">日本語</option>
                        <option value="jv">Basa Jawa</option>
                        <option value="kn">ಕನ್ನಡ</option>
                        <option value="kk">Қазақ</option>
                        <option value="km">ខ្មែរ</option>
                        <option value="ko">한국어</option>
                        <option value="ku">Kurdî</option>
                        <option value="ky">Кыргызча</option>
                        <option value="lo">ລາວ</option>
                        <option value="la">Latina</option>
                        <option value="lv">Latviešu</option>
                        <option value="lt">Lietuvių</option>
                        <option value="lb">Lëtzebuergesch</option>
                        <option value="mk">Македонски</option>
                        <option value="mg">Malagasy</option>
                        <option value="ms">Bahasa Melayu</option>
                        <option value="ml">മലയാളം</option>
                        <option value="mt">Malti</option>
                        <option value="mi">Māori</option>
                        <option value="mr">मराठी</option>
                        <option value="mn">Монгол</option>
                        <option value="ne">नेपाली</option>
                        <option value="no">Norsk</option>
                        <option value="or">ଓଡ଼ିଆ</option>
                        <option value="ps">پښتو</option>
                        <option value="fa">فارسی</option>
                        <option value="pl">Polski</option>
                        <option value="pt">Português</option>
                        <option value="pa">ਪੰਜਾਬੀ</option>
                        <option value="ro">Română</option>
                        <option value="ru">Русский</option>
                        <option value="sa">संस्कृतम्</option>
                        <option value="sr">Српски</option>
                        <option value="sd">سنڌي</option>
                        <option value="si">සිංහල</option>
                        <option value="sk">Slovenčina</option>
                        <option value="sl">Slovenščina</option>
                        <option value="so">Soomaali</option>
                        <option value="es">Español</option>
                        <option value="su">Basa Sunda</option>
                        <option value="sw">Kiswahili</option>
                        <option value="sv">Svenska</option>
                        <option value="tg">Тоҷикӣ</option>
                        <option value="ta">தமிழ்</option>
                        <option value="tt">Татарча</option>
                        <option value="te">తెలుగు</option>
                        <option value="th">ไทย</option>
                        <option value="tr">Türkçe</option>
                        <option value="tk">Türkmen</option>
                        <option value="uk">Українська</option>
                        <option value="ur">اردو</option>
                        <option value="uz">Oʻzbek</option>
                        <option value="vi">Tiếng Việt</option>
                        <option value="cy">Cymraeg</option>
                        <option value="xh">isiXhosa</option>
                        <option value="yi">ייִדיש</option>
                        <option value="yo">Yorùbá</option>
                        <option value="zu">isiZulu</option>
                    </x-forms.input>

                    <x-forms.input
                        class="capitalize"
                        type="select"
                        x-model="filters.use_cases"
                        @change="reload()"
                    >
                        <option value="">{{ __('Any use case') }}</option>
                        <option value="narrative_story">{{ __('Narrative & Story') }}</option>
                        <option value="conversational">{{ __('Conversational') }}</option>
                        <option value="characters_animation">{{ __('Characters & Animation') }}</option>
                        <option value="social_media">{{ __('Social Media') }}</option>
                        <option value="entertainment_tv">{{ __('Entertainment & TV') }}</option>
                        <option value="advertisement">{{ __('Advertisement') }}</option>
                        <option value="informative_educational">{{ __('Informative & Educational') }}</option>
                    </x-forms.input>
                </div>

                <div
                    class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3"
                    x-show="!loading || voices.length"
                >
                    <template
                        x-for="voice in voices"
                        :key="voice.public_owner_id + ':' + voice.voice_id"
                    >
                        <x-card
                            class="flex flex-col"
                            class:body="flex flex-col"
                            size="none"
                        >
                            <div class="flex items-center gap-2 border-b px-3 py-2.5">
                                <template x-if="voice.image_url">
                                    <img
                                        class="size-12 shrink-0 rounded-full object-cover object-center"
                                        :src="voice.image_url"
                                        :alt="voice.name"
                                    />
                                </template>

                                <div class="min-w-0 flex-1">
                                    <p
                                        class="m-0 truncate text-sm font-semibold capitalize"
                                        x-text="voice.name"
										:title="voice.name"
                                    ></p>
                                    <p
                                        class="m-0 truncate text-2xs opacity-60 capitalize"
                                        x-text="[voice.gender, voice.age, voice.accent, voice.locale].filter(Boolean).join(' · ')"
										:title="[voice.gender, voice.age, voice.accent, voice.locale].filter(Boolean).join(' · ')"
                                    ></p>
                                </div>

                                <template x-if="voice.preview_url">
                                    <x-button
                                        class="size-10 shrink-0 p-0 text-primary"
                                        type="button"
                                        variant="outline"
                                        size="none"
                                        hoverVariant="primary"
                                        :title="__('Preview')"
                                        @click="preview(voice)"
                                    >
                                        <x-tabler-player-play
                                            class="size-4 fill-current"
                                            x-show="previewingId !== voice.voice_id"
                                        />
                                        <x-tabler-volume
                                            class="size-4"
                                            x-cloak
                                            x-show="previewingId === voice.voice_id"
                                        />
                                    </x-button>
                                </template>
                            </div>

                            <div class="flex grow flex-col p-4">
                                <p
                                    class="mb-3 line-clamp-3 text-2xs"
                                    x-show="voice.description?.trim()"
                                    x-text="voice.description || ''"
                                ></p>

                                <template x-if="!addedIds.includes(voice.voice_id)">
                                    <x-button
                                        class="mt-auto w-full text-2xs"
                                        type="button"
                                        variant="ghost-shadow"
                                        hover-variant="primary"
                                        ::disabled="addingId === voice.voice_id"
                                        @click="addVoice(voice)"
                                    >
                                        <x-tabler-plus
                                            class="size-4"
                                            x-show="addingId !== voice.voice_id"
                                        />
                                        <x-tabler-loader-2
                                            class="size-4 animate-spin"
                                            x-cloak
                                            x-show="addingId === voice.voice_id"
                                        />
                                        <span x-text="addingId === voice.voice_id ? '{{ __('Adding...') }}' : '{{ __('Add to allowed voices') }}'"></span>
                                    </x-button>
                                </template>

                                <template x-if="addedIds.includes(voice.voice_id)">
                                    <x-button
                                        class="mt-auto w-full text-2xs"
                                        type="button"
                                        variant="ghost-shadow"
                                        hoverVariant="danger"
                                        ::disabled="removingId === voice.voice_id"
                                        @click="removeVoice(voice)"
                                    >
                                        <x-tabler-check
                                            class="size-4"
                                            x-show="removingId !== voice.voice_id"
                                        />
                                        <x-tabler-loader-2
                                            class="size-4 animate-spin"
                                            x-cloak
                                            x-show="removingId === voice.voice_id"
                                        />
                                        <span x-text="removingId === voice.voice_id ? '{{ __('Removing...') }}' : '{{ __('Allowed · Remove') }}'"></span>
                                    </x-button>
                                </template>
                            </div>
                        </x-card>
                    </template>
                </div>

                <p
                    class="m-0 px-4 py-10 text-center text-2xs opacity-60"
                    x-show="!loading && voices.length === 0"
                    x-cloak
                >
                    {{ __('No voices match your filters.') }}
                </p>

                <p
                    class="m-0 flex items-center justify-center gap-1 p-5 text-center text-2xs font-medium"
                    x-show="loading"
                >
                    <x-tabler-loader-2 class="size-4 animate-spin" />
                    {{ __('Loading') }}
                </p>

                <div
                    class="h-1"
                    x-show="hasMore"
                    x-intersect="loadMore()"
                ></div>

                <audio
                    class="hidden"
                    x-ref="previewAudio"
                    @ended="previewingId = null"
                ></audio>
            </div>
        </x-slot:modal>
    </x-modal>
</div>

@pushOnce('script')
    <script>
        window.__elevenLibraryAddedIds = window.__elevenLibraryAddedIds || {};
        window.__elevenLibraryAddedIds[@json($modalId)] = @json($addedIds);

        document.addEventListener('alpine:init', () => {
            const initialAddedIds = (window.__elevenLibraryAddedIds && window.__elevenLibraryAddedIds['{{ $modalId }}']) || [];

            Alpine.data('elevenlabsSharedLibrary', () => ({
                indexUrl: '{{ route($indexRoute) }}',
                addUrl: '{{ route($addRoute) }}',
                removeUrl: '{{ route($removeRoute) }}',
                csrf: '{{ csrf_token() }}',
                modalId: '{{ $modalId }}',
                filters: {
                    search: '',
                    gender: '',
                    age: '',
                    language: '',
                    use_cases: ''
                },
                voices: [],
                page: 1,
                pageSize: 24,
                hasMore: false,
                loading: false,
                addingId: null,
                removingId: null,
                addedIds: Array.isArray(initialAddedIds) ? [...initialAddedIds] : [],
                previewingId: null,
                initialized: false,

                get totalLabel() {
                    if (this.loading && !this.voices.length) return '';
                    return `${this.voices.length} ${this.hasMore ? '+' : ''} {{ __('voices') }}`;
                },

                getModalData() {
                    if (!this.modalData) {
                        this.modalData = Alpine.$data(document.querySelector('#{{ $modalId }}'));
                    }

                    return this.modalData;
                },

                formatNumber(value) {
                    const n = Number(value || 0);
                    if (!n) return '0';
                    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1).replace(/\.0$/, '') + 'M';
                    if (n >= 1_000) return (n / 1_000).toFixed(1).replace(/\.0$/, '') + 'k';
                    return String(n);
                },

                formatRate(rate) {
                    if (rate === null || rate === undefined) return '';
                    const n = Number(rate);
                    if (Number.isNaN(n)) return String(rate);
                    return n.toString() + '×';
                },

                formatDate(unix) {
                    if (!unix) return '';
                    try {
                        return new Date(unix * 1000).toLocaleDateString();
                    } catch (e) {
                        return '';
                    }
                },

                buildQuery(extra = {}) {
                    const params = new URLSearchParams();
                    params.set('page_size', this.pageSize);
                    params.set('page', extra.page ?? this.page);
                    Object.entries(this.filters).forEach(([k, v]) => {
                        if (v !== '' && v !== null && v !== undefined) params.set(k, v);
                    });
                    return params.toString();
                },

                async reload() {
                    this.page = 1;
                    this.voices = [];
                    await this.fetchPage();
                },

                async loadMore() {
                    if (this.loading || !this.hasMore) return;
                    this.page += 1;
                    await this.fetchPage();
                },

                async fetchPage() {
                    this.loading = true;
                    try {
                        const res = await fetch(`${this.indexUrl}?${this.buildQuery()}`, {
                            headers: {
                                'Accept': 'application/json',
                                'X-Requested-With': 'XMLHttpRequest'
                            },
                        });
                        const json = await res.json();
                        if (json.status !== 'success') {
                            window.toastr && toastr.error(json.message || 'Failed to load voices');
                            return;
                        }
                        const data = json.resData || {};
                        const list = data.voices || [];
                        this.voices = this.page === 1 ? list : [...this.voices, ...list];
                        this.hasMore = !!data.has_more;
                    } catch (e) {
                        window.toastr && toastr.error('Failed to load voices');
                    } finally {
                        this.loading = false;
                    }
                },

                preview(voice) {
                    if (!voice.preview_url) return;
                    const audio = this.$refs.previewAudio;
                    if (this.previewingId === voice.voice_id) {
                        audio.pause();
                        this.previewingId = null;
                        return;
                    }
                    audio.src = voice.preview_url;
                    audio.play();
                    this.previewingId = voice.voice_id;
                },

                async addVoice(voice) {
                    if (this.addingId) return;
                    this.addingId = voice.voice_id;
                    try {
                        const res = await fetch(this.addUrl, {
                            method: 'POST',
                            headers: {
                                'Accept': 'application/json',
                                'Content-Type': 'application/json',
                                'X-Requested-With': 'XMLHttpRequest',
                                'X-CSRF-TOKEN': this.csrf,
                            },
                            body: JSON.stringify({
                                public_user_id: voice.public_owner_id,
                                voice_id: voice.voice_id,
                                name: voice.name,
                                preview_url: voice.preview_url || '',
                                language: voice.language || (voice.locale ? String(voice.locale).split('-')[0] : ''),
                            }),
                        });
                        const json = await res.json();
                        if (json.status !== 'success') {
                            window.toastr && toastr.error(json.message || 'Failed to add voice');
                            return;
                        }
                        const newId = (json.voice && json.voice.voice_id) || voice.voice_id;
                        if (!this.addedIds.includes(newId)) this.addedIds.push(newId);
                        window.toastr && toastr.success('{{ __('Voice added to allowed voices.') }}');
                    } catch (e) {
                        window.toastr && toastr.error('Failed to add voice');
                    } finally {
                        this.addingId = null;
                    }
                },

                async removeVoice(voice) {
                    if (this.removingId) return;
                    this.removingId = voice.voice_id;
                    try {
                        const res = await fetch(this.removeUrl, {
                            method: 'POST',
                            headers: {
                                'Accept': 'application/json',
                                'Content-Type': 'application/json',
                                'X-Requested-With': 'XMLHttpRequest',
                                'X-CSRF-TOKEN': this.csrf,
                            },
                            body: JSON.stringify({
                                voice_id: voice.voice_id
                            }),
                        });
                        const json = await res.json();
                        if (json.status !== 'success') {
                            window.toastr && toastr.error(json.message || 'Failed to remove voice');
                            return;
                        }
                        this.addedIds = this.addedIds.filter(id => id !== voice.voice_id);
                        window.toastr && toastr.success('{{ __('Voice removed from allowed voices.') }}');
                    } catch (e) {
                        window.toastr && toastr.error('Failed to remove voice');
                    } finally {
                        this.removingId = null;
                    }
                },

                toggleModal(state) {
                    if (state && !this.initialized) {
                        this.initialized = true;
                        this.reload();
                    }

                    this.getModalData().toggleModal(state);
                }
            }));
        });
    </script>
@endpushOnce
