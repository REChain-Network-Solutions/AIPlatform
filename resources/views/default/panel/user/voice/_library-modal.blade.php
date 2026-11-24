@php
    $uid = $uid ?? uniqid();
    $modalId = 'user-voice-library-modal-' . $uid;
    $triggerLabel = $triggerLabel ?? __('Browse Voice Library');
    $triggerClass = @twMerge('h-11 min-h-10 shrink-0 gap-2 overflow-hidden !rounded-input bg-background px-4 font-normal capitalize hover:shadow-none sm:text-2xs', $triggerClass ?? '');
@endphp

<div
    class="contents"
    x-data="userVoiceLibrary"
>
    <x-button
        @class($triggerClass)
        type="button"
        variant="outline"
        size="none"
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
                    x-model.debounce.200ms="filters.search"
                />

                <div class="mb-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <x-forms.input
                        class="capitalize"
                        type="select"
                        x-model="filters.language"
                    >
                        <option value="">{{ __('Any language') }}</option>
                        <template
                            x-for="opt in facets.language"
                            :key="opt.value"
                        >
                            <option
                                :value="opt.value"
                                x-text="opt.label"
                            ></option>
                        </template>
                    </x-forms.input>

                    <x-forms.input
                        class="capitalize"
                        type="select"
                        x-model="filters.gender"
                    >
                        <option value="">{{ __('Any gender') }}</option>
                        <template
                            x-for="opt in facets.gender"
                            :key="opt"
                        >
                            <option
                                :value="opt"
                                x-text="opt"
                            ></option>
                        </template>
                    </x-forms.input>
                </div>

                <div
                    class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3"
                    x-show="filteredVoices.length"
                >
                    <template
                        x-for="voice in filteredVoices"
                        :key="voice.platform + ':' + voice.value"
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
                                        class="mb-0.5 truncate text-2xs font-medium capitalize"
                                        x-text="voice.name"
                                        :title="voice.name"
                                    ></p>
                                    <p
                                        class="m-0 truncate text-2xs font-medium capitalize opacity-50"
                                        x-text="[voice.gender, voice.age, voice.accent, displayLanguageFor(voice)].filter(Boolean).join(' · ')"
                                        :title="[voice.gender, voice.age, voice.accent, displayLanguageFor(voice)].filter(Boolean).join(' · ')"
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
                                            class="size-4"
                                            x-show="previewingKey !== keyFor(voice)"
                                        />
                                        <x-tabler-volume
                                            class="size-4"
                                            x-cloak
                                            x-show="previewingKey === keyFor(voice)"
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

                                <x-button
                                    class="mt-auto w-full text-2xs"
                                    type="button"
                                    variant="ghost-shadow"
                                    hover-variant="primary"
                                    @click="selectVoice(voice)"
                                >
                                    {{ __('Select') }}
                                </x-button>
                            </div>
                        </x-card>
                    </template>
                </div>

                <p
                    class="m-0 px-4 py-10 text-center text-2xs opacity-60"
                    x-show="!filteredVoices.length"
                    x-cloak
                >
                    {{ __('No voices match your filters.') }}
                </p>

                <audio
                    class="hidden"
                    x-ref="previewAudio"
                    @ended="previewingKey = null"
                ></audio>
            </div>
        </x-slot:mod>
    </x-modal>
</div>

@pushOnce('script')
    <script>
        window.__ttsGoogleEnabled = @json((bool) ($settings_two->feature_tts_google ?? false));
        window.__ttsOpenAIEnabled = @json((bool) ($settings_two->feature_tts_openai ?? false));
        window.__ttsElevenLabsEnabled = @json((bool) ($settings_two->feature_tts_elevenlabs ?? false));

        document.addEventListener('alpine:init', () => {
            Alpine.data('userVoiceLibrary', () => ({
                filters: {
                    search: '',
                    language: '',
                    gender: ''
                },
                previewingKey: null,
                modalData: null,

                keyFor(v) {
                    return `${v.platform}:${v.value}`;
                },

                languageLabel(code) {
                    const select = document.getElementById('languages');
                    if (!select) return code;
                    const opt = Array.from(select.options).find((o) => o.value === code);
                    return opt ? (opt.textContent || '').trim() : code;
                },

                // When the user has picked a specific language, show that label on each card.
                // ElevenLabs voices are stored prefix-only ("en") and inflated to every "en-*"
                // variant for filter membership, so without this the card would show a baked-in
                // arbitrary region (e.g. "English (Australia)") that contradicts the filter.
                displayLanguageFor(voice) {
                    if (this.filters.language && (voice.languages || []).includes(this.filters.language)) {
                        return this.languageLabel(this.filters.language);
                    }
                    return voice.languageLabel || '';
                },

                get voices() {
                    const list = [];
                    const seen = new Map();
                    const googleEnabled = !!window.__ttsGoogleEnabled;
                    const openaiEnabled = !!window.__ttsOpenAIEnabled;
                    const elevenEnabled = !!window.__ttsElevenLabsEnabled;

                    const pushUnique = (entry) => {
                        const key = `${entry.platform}:${entry.value}`;
                        if (seen.has(key)) return;
                        seen.set(key, true);
                        list.push(entry);
                    };

                    // Google/Azure voices: one entry per voice value. Carry every language it supports.
                    if (googleEnabled && typeof voicesData !== 'undefined' && voicesData) {
                        const googleByValue = new Map();
                        Object.keys(voicesData).forEach((lang) => {
                            const arr = voicesData[lang];
                            if (!Array.isArray(arr)) return;
                            arr.forEach((v) => {
                                if (!v || !v.value) return;
                                const existing = googleByValue.get(v.value);
                                if (existing) {
                                    if (!existing.languages.includes(lang)) existing.languages.push(lang);
                                    return;
                                }
                                googleByValue.set(v.value, {
                                    platform: 'google',
                                    value: v.value,
                                    name: v.label,
                                    language: lang,
                                    languageLabel: this.languageLabel(lang),
                                    languages: [lang],
                                    preview_url: null,
                                    gender: null,
                                    age: null,
                                    accent: null,
                                    use_case: null,
                                    descriptive: null,
                                    image_url: null,
                                    description: null,
                                });
                            });
                        });
                        googleByValue.forEach((v) => pushUnique(v));
                    }

                    // OpenAI voices: one entry per voice value. Languages = full allowed list.
                    if (openaiEnabled &&
                        typeof openaiVoiceData !== 'undefined' && Array.isArray(openaiVoiceData) &&
                        typeof allowedOpenAIList !== 'undefined' && Array.isArray(allowedOpenAIList)) {
                        openaiVoiceData.forEach((v) => {
                            if (!v || !v.value) return;
                            const primary = allowedOpenAIList[0] || '';
                            pushUnique({
                                platform: 'openai',
                                value: v.value,
                                name: `${v.label} (OpenAI)`,
                                language: primary,
                                languageLabel: primary ? this.languageLabel(primary) : '',
                                languages: [...allowedOpenAIList],
                                preview_url: null,
                                gender: null,
                                age: null,
                                accent: null,
                                use_case: null,
                                descriptive: null,
                                image_url: null,
                                description: null,
                            });
                        });
                    }

                    // ElevenLabs voices: one entry per voice_id. Languages = allowed langs whose prefix matches.
                    if (elevenEnabled &&
                        typeof elevenLabsVoices !== 'undefined' && Array.isArray(elevenLabsVoices) &&
                        typeof allowedElevenLabsList !== 'undefined' && Array.isArray(allowedElevenLabsList)) {
                        const langSelect = document.getElementById('languages');
                        const allLangValues = langSelect ?
                            Array.from(langSelect.options).map((o) => o.value) : [];

                        elevenLabsVoices.forEach((v) => {
                            if (!v || !v.voice_id) return;
                            const voiceLang = String(v.language || '').toLowerCase();
                            const voicePrefix = voiceLang.split(/[-_]/)[0];

                            const matchingLangs = voicePrefix ?
                                allLangValues.filter((lv) => {
                                    if (!allowedElevenLabsList.includes(lv)) return false;
                                    const pref = String(lv).toLowerCase().split(/[-_]/)[0];
                                    return pref === voicePrefix;
                                }) : [];

                            // Use the voice's actual prefix as its primary language so the
                            // card label doesn't lie about a specific region. Filter membership
                            // still uses the inflated matchingLangs list below.
                            const primary = voicePrefix || '';

                            pushUnique({
                                platform: 'elevenlabs',
                                value: v.voice_id,
                                name: v.name,
                                language: primary,
                                languageLabel: primary ? (this.languageLabel(primary) || '') : '',
                                languages: matchingLangs.length ? matchingLangs : (voicePrefix ? [voicePrefix] : []),
                                preview_url: v.preview_url || null,
                                gender: v.gender || null,
                                age: v.age || null,
                                accent: v.accent || null,
                                use_case: v.use_case || null,
                                descriptive: v.descriptive || null,
                                image_url: v.image_url || null,
                                description: v.description || null,
                            });
                        });
                    }

                    return list;
                },

                get facets() {
                    const langs = new Map();
                    const genders = new Set();
                    this.voices.forEach((v) => {
                        (v.languages || [])
                        .forEach((lang) => {
                            if (lang && !langs.has(lang)) langs.set(lang, this.languageLabel(lang) || lang);
                        });
                        if (v.gender) genders.add(v.gender);
                    });
                    return {
                        language: Array.from(langs.entries())
                            .map(([value, label]) => ({
                                value,
                                label
                            }))
                            .sort((a, b) => a.label.localeCompare(b.label)),
                        gender: Array.from(genders).sort(),
                    };
                },

                get filteredVoices() {
                    const q = (this.filters.search || '').trim().toLowerCase();
                    return this.voices.filter((v) => {
                        if (this.filters.language && !(v.languages || []).includes(this.filters.language)) return false;
                        if (this.filters.gender && v.gender !== this.filters.gender) return false;
                        if (!q) return true;
                        const hay = [v.name, v.description, v.accent, v.use_case, v.descriptive, ...(v.languages || []), v.languageLabel]
                            .filter(Boolean)
                            .join(' ')
                            .toLowerCase();
                        return hay.includes(q);
                    });
                },

                get totalLabel() {
                    return `${this.filteredVoices.length} {{ __('voices') }}`;
                },

                getModalData() {
                    if (!this.modalData) {
                        this.modalData = Alpine.$data(document.querySelector('#{{ $modalId }}'));
                    }

                    return this.modalData;
                },

                preview(voice) {
                    if (!voice.preview_url) return;
                    const audio = this.$refs.previewAudio;
                    const key = this.keyFor(voice);
                    if (this.previewingKey === key) {
                        audio.pause();
                        this.previewingKey = null;
                        return;
                    }
                    audio.src = voice.preview_url;
                    audio.play();
                    this.previewingKey = key;
                },

                fireChange(el) {
                    if (!el) return;
                    // Fire both jQuery (for populateVoiceSelect/populatePaceSelect) and native
                    // (for the Alpine voiceoverPickers addEventListener handlers).
                    if (window.jQuery) {
                        window.jQuery(el).trigger('change');
                    }
                    el.dispatchEvent(new Event('change', {
                        bubbles: true
                    }));
                },

                pickLanguageFor(voice) {
                    const compat = voice.languages || [];
                    if (!compat.length) return voice.language || '';
                    if (this.filters.language && compat.includes(this.filters.language)) {
                        return this.filters.language;
                    }
                    const languagesEl = document.getElementById('languages');
                    if (languagesEl && compat.includes(languagesEl.value)) {
                        return languagesEl.value;
                    }
                    return compat[0];
                },

                selectVoice(voice) {
                    const languagesEl = document.getElementById('languages');
                    const voiceEl = document.getElementById('voice');
                    const targetLang = this.pickLanguageFor(voice);

                    if (languagesEl && targetLang) {
                        const has = Array.from(languagesEl.options).some((o) => o.value === targetLang);
                        if (has && languagesEl.value !== targetLang) {
                            languagesEl.value = targetLang;
                            this.fireChange(languagesEl);
                        }
                    }

                    const applyVoice = () => {
                        if (!voiceEl) return;
                        const exists = Array.from(voiceEl.options).some((o) => o.value === voice.value);
                        if (!exists) {
                            const opt = document.createElement('option');
                            opt.value = voice.value;
                            opt.textContent = voice.name;
                            opt.setAttribute('platform', voice.platform);
                            opt.setAttribute('name', voice.name);
                            voiceEl.appendChild(opt);
                        }
                        voiceEl.value = voice.value;
                        this.fireChange(voiceEl);
                    };

                    // populateVoiceSelect runs on language change and rebuilds #voice options;
                    // wait a tick before setting the voice so the option exists.
                    setTimeout(applyVoice, 80);

                    this.toggleModal(false);
                    window.toastr && toastr.success('{{ __('Voice selected.') }}');
                },

                toggleModal(state) {
                    this.getModalData().toggleModal(state);
                }
            }));
        });
    </script>
@endPushOnce
