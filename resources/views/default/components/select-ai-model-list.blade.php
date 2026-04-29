@php
    use App\Domains\Engine\Enums\EngineEnum;
    use App\Domains\Entity\Models\Entity;
    use App\Domains\Entity\Enums\EntityEnum;
    use App\Extensions\OpenRouter\System\Enums\OpenRouterEngine;
    use App\Helpers\Classes\MarketplaceHelper;

    $defaultEngine = EngineEnum::fromSlug(setting('default_ai_engine', EngineEnum::OPEN_AI->slug()));
    $defaultModel = $defaultEngine->getDefaultWordModel($setting);

    $fullModels = [$defaultModel];
    if (Entity::planModels()->count() > 0) {
        $planModels = Entity::planModels()
            ->filter(function ($model) {
                return !empty($model->key);
            })
            ->pluck('key')
            ->toArray();

        $fullModels = array_merge($fullModels, $planModels);
    }

    if ((int) setting('open_router_status') === 1) {
        $openRouterModels = OpenRouterEngine::cases();

        $fullModels = array_merge($fullModels, $openRouterModels);
    }

    $fullModels = collect($fullModels)->unique('value')->values();
    $activeModels = collect($fullModels)
        ->filter(function ($model) {
            $driver = \App\Domains\Entity\Facades\Entity::driver(EntityEnum::tryFrom($model?->value));
            return $driver->isUnlimitedCredit() || $driver->creditBalance() > 0;
        })
        ->values();
    $defaultDriver = \App\Domains\Entity\Facades\Entity::driver(EntityEnum::tryFrom($defaultModel?->value));
    $selectedModel = $defaultDriver;
    if (!$defaultDriver->isUnlimitedCredit() && $defaultDriver->creditBalance() <= 0) {
        foreach ($fullModels as $model) {
            $driver = \App\Domains\Entity\Facades\Entity::driver(EntityEnum::tryFrom($model?->value));
            if ($driver->isUnlimitedCredit() || $driver->creditBalance() > 0) {
                $selectedModel = $driver;
                break;
            }
        }
    }

    $currentUrl = url()->current();
    $previousUrl = url()->previous();
    $is_chat_pro =
        \App\Helpers\Classes\MarketplaceHelper::isRegistered('ai-chat-pro') &&
        (route('dashboard.user.openai.chat.pro.index') === $currentUrl ||
            route('chat.pro') === $currentUrl ||
            route('dashboard.user.openai.chat.pro.index') === $previousUrl ||
            route('chat.pro') === $previousUrl);

    $isMultiModelExtensionEnabled = $is_chat_pro && MarketplaceHelper::isRegistered('multi-model') && setting('ai_chat_pro_multi_model_feature', '1') === '1';
    $isCouncilExtensionEnabled = $is_chat_pro && MarketplaceHelper::isRegistered('model-council') && setting('ai_chat_pro_model_council_feature', '1') === '1';
    $canMultiModelSelect = $isMultiModelExtensionEnabled && (auth()?->user()?->activePlan()?->multi_model_support ?? false || auth()?->user()?->isAdmin());
    $canCouncilSelect = $isCouncilExtensionEnabled && (auth()?->user()?->activePlan()?->model_council_support ?? false || auth()?->user()?->isAdmin());
    $councilMaxModels = max(2, min(5, (int) setting('ai_chat_pro_council_max_models', 5)));
    $supportsMultiSelection = $isMultiModelExtensionEnabled || $isCouncilExtensionEnabled;
@endphp

<div x-data="modelList">
    <x-modal
        class:modal="select-ai-model-modal"
        class:modal-body="max-sm:p-3"
        class:modal-head="sticky top-0 z-[100] shrink-0 flex-nowrap gap-1 bg-background p-4 max-lg:items-start"
        class:modal-content="mx-5 container"
        class:close-btn="lg:!hidden"
        id="openRouterModel"
        disable-focus="true"
    >
        <x-slot:trigger
            variant="ghost-shadow"
        >
            <svg
                width="18"
                height="16"
                viewBox="0 0 18 16"
                fill="currentColor"
                xmlns="http://www.w3.org/2000/svg"
            >
                <path
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                    d="M17.007 6.15216L15.615 6.43416V6.43616C14.9445 6.57221 14.329 6.90274 13.8453 7.38647C13.3616 7.87021 13.031 8.48572 12.895 9.15617L12.613 10.5482C12.5832 10.6852 12.5075 10.8079 12.3983 10.8958C12.2892 10.9838 12.1532 11.0318 12.013 11.0318C11.8728 11.0318 11.7368 10.9838 11.6277 10.8958C11.5185 10.8079 11.4427 10.6852 11.413 10.5482L11.131 9.15617C10.9952 8.48561 10.6647 7.86997 10.181 7.38619C9.69718 6.90241 9.08153 6.57197 8.41098 6.43616L7.01898 6.15416C6.88017 6.12574 6.75543 6.05026 6.66585 5.94048C6.57627 5.8307 6.52734 5.69336 6.52734 5.55166C6.52734 5.40997 6.57627 5.27263 6.66585 5.16285C6.75543 5.05307 6.88017 4.97759 7.01898 4.94916L8.41098 4.66716C9.08148 4.53124 9.69706 4.20076 10.1808 3.717C10.6646 3.23324 10.9951 2.61766 11.131 1.94716L11.413 0.555164C11.4427 0.418164 11.5185 0.295476 11.6277 0.207494C11.7368 0.119511 11.8728 0.0715332 12.013 0.0715332C12.1532 0.0715332 12.2892 0.119511 12.3983 0.207494C12.5075 0.295476 12.5832 0.418164 12.613 0.555164L12.895 1.94716C13.031 2.61761 13.3616 3.23312 13.8453 3.71686C14.329 4.20059 14.9445 4.53112 15.615 4.66716L17.007 4.94716C17.1458 4.97559 17.2705 5.05107 17.3601 5.16085C17.4497 5.27063 17.4986 5.40797 17.4986 5.54966C17.4986 5.69136 17.4497 5.8287 17.3601 5.93848C17.2705 6.04826 17.1458 6.12374 17.007 6.15216ZM6.82915 13.2051L6.45115 13.2821C5.98493 13.3767 5.55688 13.6054 5.22041 13.9417C4.88394 14.278 4.65395 14.706 4.55915 15.1721L4.48215 15.5501C4.46155 15.6566 4.40452 15.7526 4.32086 15.8217C4.2372 15.8907 4.13211 15.9284 4.02365 15.9284C3.91518 15.9284 3.8101 15.8907 3.72644 15.8217C3.64278 15.7526 3.58575 15.6566 3.56515 15.5501L3.48815 15.1721C3.39347 14.7059 3.16352 14.2779 2.82703 13.9416C2.49054 13.6053 2.06242 13.3756 1.59615 13.2811L1.21815 13.2041C1.11166 13.1835 1.01566 13.1265 0.946629 13.0428C0.877599 12.9592 0.839844 12.8541 0.839844 12.7456C0.839844 12.6372 0.877599 12.5321 0.946629 12.4484C1.01566 12.3648 1.11166 12.3077 1.21815 12.2871L1.59615 12.2101C2.06256 12.1156 2.49077 11.8858 2.82727 11.5493C3.16378 11.2128 3.39364 10.7845 3.48815 10.3181L3.56515 9.94013C3.58575 9.83364 3.64278 9.73764 3.72644 9.66861C3.8101 9.59958 3.91518 9.56183 4.02365 9.56183C4.13211 9.56183 4.2372 9.59958 4.32086 9.66861C4.40452 9.73764 4.46155 9.83364 4.48215 9.94013L4.55915 10.3181C4.6536 10.7847 4.88343 11.213 5.21992 11.5497C5.55642 11.8864 5.98466 12.1164 6.45115 12.2111L6.82915 12.2881C6.93564 12.3087 7.03164 12.3658 7.10067 12.4494C7.1697 12.5331 7.20745 12.6382 7.20745 12.7466C7.20745 12.8551 7.1697 12.9602 7.10067 13.0438C7.03164 13.1275 6.93564 13.1845 6.82915 13.2051Z"
                />
            </svg>
            <span>
                <span
                    x-text="selectedModelLabel && selectedModelLabel.length > 20
							 ? selectedModelLabel.slice(0, 20) + '...'
							 : (selectedModelLabel || '@lang('None')')"
                    :title="selectedModelLabel || '@lang('None')'"
                >
                    {{ Str::limit($selectedModel->model()?->selected_title ?? $selectedModel->enum()?->value, 20) }}
                </span>
            </span>

        </x-slot:trigger>

        <x-slot:head-content>
            <div class="flex w-full flex-col gap-y-1 sm:grow">
                <div class="flex flex-wrap items-center justify-between gap-y-3 lg:flex-nowrap">
                    <div class="shrink-0">
                        <h4 class="mb-0 whitespace-nowrap">
                            @lang('AI Models')
                        </h4>
                        <template x-if="selectedModels.length === 0">
                            <p class="mb-0 text-2xs font-normal text-foreground">
                                @if ($isMultiModelExtensionEnabled || $isCouncilExtensionEnabled)
                                    @lang('Choose multiple AI models that best suit your needs.')
                                @else
                                    @lang('Choose the AI model that best suits your needs.')
                                @endif
                            </p>
                        </template>
                        @if ($isMultiModelExtensionEnabled)
                            <p class="tmp-alert hidden text-2xs text-blue-400">
                                <x-tabler-info-circle class="me-1 inline size-4" />
                                @lang('If you want to select multiple models, please upgrade your plan.')
                            </p>
                        @endif
                        @if ($isCouncilExtensionEnabled)
                            <p class="tmp-alert-council hidden text-2xs text-blue-400">
                                <x-tabler-info-circle class="me-1 inline size-4" />
                                @lang('Model Council is not included in your current plan.')
                            </p>
                        @endif
                    </div>

                    <div class="w-full lg:ms-auto">
                        <div class="flex grow flex-wrap items-center justify-between gap-2 lg:justify-end">
                            @if ($isCouncilExtensionEnabled)
                                <x-forms.input
                                    class="border-foreground/10 dark:checked:border-primary dark:checked:bg-primary"
                                    type="checkbox"
                                    name="council_mode"
                                    :label="__('Council Mode')"
                                    x-model="councilMode"
                                    switcher
                                    @change="toggleCouncilMode()"
                                />
                            @endif

                            <form
                                class="max-sm:w-full"
                                action="#"
                            >
                                <x-forms.input
                                    class="max-h-9 rounded-full bg-clay ps-8 sm:min-w-64"
                                    type="search"
                                    placeholder="{{ __('Search model') }}"
                                    @input="searchString = $event.target.value"
                                >
                                    <x-slot:icon>
                                        <x-tabler-search class="absolute start-3 top-1/2 size-4 -translate-y-1/2" />
                                    </x-slot:icon>
                                </x-forms.input>
                            </form>
                        </div>
                    </div>
                </div>

                <template x-if="selectedModels.length > 0">
                    <div class="flex flex-nowrap items-center gap-1.5 overflow-x-auto">
                        <span class="shrink-0 text-2xs font-normal text-foreground">@lang('Selected Models:')</span>
                        <template
                            x-for="(model, index) in selectedModels"
                            :key="model.value"
                        >
                            <x-button
                                class="shrink-0 whitespace-nowrap !py-1 !ps-2.5 pe-1.5 !text-2xs hover:transform-none"
                                hover-variant="secondary"
                                variant="outline"
                                size="none"
                                @click.prevent.stop="updateSelectedModels(model)"
                            >
                                <span
                                    class="whitespace-nowrap"
                                    x-text="model.label?.length > 10 ? model.label.slice(0, 10) + '...' : model.label"
                                ></span>
                                <span class="inline-grid size-5 place-items-center rounded-full text-foreground/50 transition group-hover:bg-red-500 group-hover:text-white">
                                    <x-tabler-x class="size-3.5" />
                                </span>
                            </x-button>
                        </template>
                    </div>
                </template>
            </div>
        </x-slot:head-content>

        <x-slot:modal>
            <div class="min-w-fit">
                <form
                    action="#"
                    @submit.prevent="saveChanges(); modalOpen = false;"
                >
                    <div class="flex flex-col items-end">
                        <div class="grid w-full grid-cols-1 gap-7 md:grid-cols-2 lg:grid-cols-3">
                            @foreach ($fullModels as $engine)
                                @php
                                    $model = EntityEnum::fromSlug($engine?->value);
                                    $driver = \App\Domains\Entity\Facades\Entity::driver($model);
                                    $is_inactive = $driver->creditBalance() <= 0 && !$driver->isUnlimitedCredit();
                                @endphp

                                <x-card
                                    class:body="md:p-7 p-5 static"
                                    data-model-value="{{ $model?->value }}"
                                    data-model-label="{{ $driver->model()?->selected_title ?? $model?->value }}"
                                    @class([
                                        'lqd-model-card cursor-pointer relative data-[selected]:outline data-[selected]:outline-[3px] data-[selected]:outline-secondary [&.inactive]:pointer-events-none [&.inactive]:opacity-50',
                                        'inactive' => $is_inactive,
                                    ])
                                    variant="outline-shadow"
                                    @click.prevent="updateSelectedModels({ value: $el.getAttribute('data-model-value'), label: $el.getAttribute('data-model-label') })"
                                    ::data-selected="selectedModels.findIndex(model => model?.value === $el.getAttribute('data-model-value')) >= 0 && {{ $is_inactive ? 'false' : 'true' }}"
                                    x-show="searchString === '' || $el.getAttribute('data-model-label').toLowerCase().includes(searchString.toLowerCase())"
                                >
                                    <div class="w-full">
                                        <div class="mb-6 flex justify-between gap-1.5">
                                            <figure
                                                class="{{ $driver->model()?->image ? 'bg-white' : '' }} inline-grid size-10 shrink-0 place-content-center rounded-full bg-heading-foreground/5"
                                            >
                                                @if ($driver->model()?->image)
                                                    <img
                                                        class="size-6 object-contain"
                                                        src="{{ asset($driver->model()?->image) }}"
                                                        alt="{{ $driver->model()?->selected_title ?? $model?->value }}"
                                                    >
                                                @else
                                                    <x-tabler-brand-openai
                                                        class="size-6"
                                                        stroke-width="1.5"
                                                    />
                                                @endif
                                            </figure>
                                            <div class="text-end">
                                                <p class="m-0 text-3xs font-medium text-heading-foreground">
                                                    @lang('Words')
                                                </p>
                                                <p class="m-0 text-3xs font-medium text-heading-foreground/50">
                                                    @if ($driver->isUnlimitedCredit())
                                                        @lang('Unlimited Credits')
                                                    @else
                                                        {{ $driver->creditBalance() > 0 ? $driver->creditBalance() : __('No credits left') }}
                                                    @endif
                                                </p>
                                            </div>
                                        </div>

                                        <div class="flex items-center justify-between">
                                            <div class="flex-1">
                                                <h4 class="mb-2">
                                                    {{ $driver->model()?->selected_title ?? $model?->value }}
                                                </h4>
                                            </div>
                                        </div>
                                    </div>

                                    @if ($isMultiModelExtensionEnabled)
                                        <div
                                            class="absolute bottom-3 end-3 inline-grid size-9 place-items-center rounded-full bg-secondary text-secondary-foreground shadow-lg shadow-black/5"
                                            x-show="selectedModels.find(model => model.value === '{{ $model?->value }}') && {{ $is_inactive ? 'false' : 'true' }}"
                                        >
                                            <x-tabler-check class="size-5" />
                                        </div>
                                    @endif
                                </x-card>
                            @endforeach
                        </div>

                        <x-forms.input
                            id="chatbot_front_model"
                            container-class="hidden"
                            name="chatbot_front_model"
                            type="select"
                            :multiple="$supportsMultiSelection"
                            x-ref="modelsSelectElement"
                            x-model="selectedModels.map(model => model?.value)"
                        >
                            <option
                                data-label="{{ __('Default Model') }}"
                                value=""
                            >
                                {{ __('Default Model') }}
                            </option>
                            @foreach ($fullModels as $model)
                                <option
                                    data-label="{{ $model?->label() ?? '' }}"
                                    value="{{ $model?->value }}"
                                >
                                    {{ $model?->label() }}
                                </option>
                            @endforeach
                        </x-forms.input>
                        <input
                            id="chatbot_council_mode"
                            name="chatbot_council_mode"
                            type="hidden"
                            :value="isCouncilEnabled() ? '1' : '0'"
                        />

                        <x-button
                            class="sticky bottom-5 mt-10 w-full backdrop-blur-lg disabled:bg-heading-foreground/30 disabled:text-header-background"
                            type="submit"
                            size="xl"
                            ::disabled="!selectedModels.length || (isCouncilEnabled() && selectedModels.length < 2)"
                        >
                            {{ __('Apply') }}
                        </x-button>
                    </div>
                </form>
            </div>
        </x-slot:modal>
    </x-modal>
</div>

@push('script')
    <script>
        (() => {
            document.addEventListener('alpine:init', () => {
                Alpine.data('modelList', () => ({
                    selectedModelLabel: '',
                    selectedModels: [],
                    councilMode: false,
                    fullModels: @json($fullModels),
                    activeModels: @json($activeModels),
                    searchString: '',
                    localStorageKey: 'selectedChatModels',
                    councilModeStorageKey: 'chatCouncilMode',
                    councilMaxModels: {{ $councilMaxModels }},
                    canMultiModelSelect: {{ $canMultiModelSelect ? 'true' : 'false' }},
                    canCouncilSelect: {{ $canCouncilSelect ? 'true' : 'false' }},
                    isMultiModelExtensionEnabled: {{ $isMultiModelExtensionEnabled ? 'true' : 'false' }},
                    isCouncilExtensionEnabled: {{ $isCouncilExtensionEnabled ? 'true' : 'false' }},
                    isAuthenticated: {{ auth()->check() ? 'true' : 'false' }},

                    getLocalStorage() {
                        const defaultModelValue = '{{ $selectedModel->enum()?->value }}';
                        const defaultModelLabel = '{{ $selectedModel->model()?->selected_title ?? $selectedModel->enum()?->value }}';
                        const localStorageLastSelectedModels = localStorage.getItem(this.localStorageKey) ??
                            `[{ "value": "${defaultModelValue}", "label": "${defaultModelLabel}" }]`;
                        const models = JSON.parse(localStorageLastSelectedModels)
                            .filter(model => this.fullModels.find(m => (typeof m === 'string' ? m : m?.value) === model?.value))
                            .filter(model => this.activeModels.find(m => (typeof m === 'string' ? m : m?.value) === model?.value));

                        let safeModels = models;

                        if (!safeModels.length) {
                            const isDefaultActive = this.activeModels.some(m => (typeof m === 'string' ? m : m?.value) === defaultModelValue);

                            if (isDefaultActive) {
                                safeModels = [{
                                    value: defaultModelValue,
                                    label: defaultModelLabel,
                                }];
                            } else if (this.activeModels.length > 0) {
                                const firstActive = this.activeModels[0];
                                const val = typeof firstActive === 'string' ? firstActive : firstActive?.value;
                                const selectEl = document.getElementById('chatbot_front_model');
                                const option = selectEl?.querySelector(`option[value="${val}"]`);
                                safeModels = [{
                                    value: val,
                                    label: option?.getAttribute('data-label') ?? val,
                                }];
                            }
                        }

                        @if ($supportsMultiSelection)
                            return safeModels;
                        @else
                            return safeModels.length ? [safeModels[0]] : [];
                        @endif
                    },
                    setLocalStorage() {
                        localStorage.setItem(this.localStorageKey, JSON.stringify(this.selectedModels));
                    },
                    getCouncilModeStorage() {
                        return localStorage.getItem(this.councilModeStorageKey) === '1';
                    },
                    setCouncilModeStorage() {
                        localStorage.setItem(this.councilModeStorageKey, this.isCouncilEnabled() ? '1' : '0');
                    },
                    isCouncilEnabled() {
                        return this.isCouncilExtensionEnabled && this.councilMode && this.canCouncilSelect;
                    },
                    showCouncilPlanAlert() {
                        const el = document.querySelector('.tmp-alert-council');
                        if (!el) return;

                        el.classList.remove('hidden');
                        setTimeout(() => {
                            el.classList.add('hidden');
                        }, 3000);
                    },
                    toggleCouncilMode() {
                        if (this.councilMode && !this.canCouncilSelect) {
                            this.councilMode = false;
                            if (this.isAuthenticated) {
                                this.showCouncilPlanAlert();
                            } else {
                                toastr.warning('{{ __('Login required to use Council Mode') }}', '{{ __('Login Required') }}');
                            }
                            return;
                        }

                        this.normalizeSelectionByMode();
                        this.updateSelectionLabel();
                        this.setLocalStorage();
                        this.setCouncilModeStorage();

                        if (this.isCouncilEnabled()) {
                            const deepResearch = Alpine.store('deepResearchStatus');
                            if (deepResearch?.status) {
                                deepResearch.status = false;
                                toastr.info('{{ __('Deep Research has been disabled. Council Mode uses multiple models simultaneously.') }}');
                            }
                            toastr.success('{{ __('Council Mode enabled') }}');
                        } else {
                            toastr.info('{{ __('Council Mode disabled') }}');
                        }
                    },
                    normalizeSelectionByMode() {
                        if (this.isCouncilEnabled()) {
                            if (this.selectedModels.length > this.councilMaxModels) {
                                this.selectedModels = this.selectedModels.slice(0, this.councilMaxModels);
                            }

                            return;
                        }

                        if (this.canMultiModelSelect) {
                            if (this.selectedModels.length > 2) {
                                this.selectedModels = this.selectedModels.slice(0, 2);
                            }

                            return;
                        }

                        this.selectedModels = this.selectedModels.length ? [this.selectedModels[0]] : [];
                    },
                    updateSelectedModels(modelObj = {
                        value: null,
                        label: null
                    }) {
                        const existingSelectedModels = this.selectedModels;
                        const existingIndex = this.selectedModels.findIndex(model => model?.value === modelObj?.value);

                        if (this.isCouncilEnabled()) {
                            if (existingIndex > -1) {
                                if (this.selectedModels.length > 2) {
                                    this.selectedModels = this.selectedModels.filter((model, index) => index !== existingIndex);
                                } else {
                                    toastr.error('{{ __('Model Council requires at least 2 selected models.') }}');
                                    this.selectedModels = existingSelectedModels;
                                }
                            } else if (this.selectedModels.length >= this.councilMaxModels) {
                                toastr.error('{{ __('You can select up to') }} ' + this.councilMaxModels + ' {{ __('models in Council mode.') }}');
                                this.selectedModels = existingSelectedModels;
                            } else {
                                this.selectedModels.push({
                                    value: modelObj?.value,
                                    label: modelObj?.label,
                                });
                            }
                        } else if (this.canMultiModelSelect) {
                            if (existingIndex > -1) {
                                if (this.selectedModels.length > 1) {
                                    this.selectedModels = this.selectedModels.filter((model, index) => index !== existingIndex);
                                }
                            } else if (this.selectedModels.length >= 2) {
                                toastr.error('{{ __('Selecting more than 2 models not allowed') }}');
                                this.selectedModels = existingSelectedModels;
                            } else {
                                this.selectedModels.push({
                                    value: modelObj?.value,
                                    label: modelObj?.label,
                                });
                            }
                        } else {
                            if (this.isCouncilExtensionEnabled && this.councilMode && !this.canCouncilSelect) {
                                this.showCouncilPlanAlert();
                                this.councilMode = false;
                            } else if (this.isMultiModelExtensionEnabled) {
                                document.querySelector('.tmp-alert')?.classList.remove('hidden');
                                setTimeout(() => {
                                    document.querySelector('.tmp-alert')?.classList.add('hidden');
                                }, 3000);
                            }

                            this.selectedModels = [modelObj];
                        }

                        this.updateSelectionLabel();
                        this.setLocalStorage();
                    },
                    updateSelectionLabel() {
                        if (this.selectedModels.length === 0) {
                            this.selectedModelLabel = '{{ __('None') }}';
                            return;
                        }

                        if (!this.isCouncilEnabled() && !this.canMultiModelSelect) {
                            this.selectedModelLabel = this.selectedModels[0]?.label ?? '{{ __('None') }}';
                            return;
                        }

                        if (this.selectedModels.length === 1) {
                            this.selectedModelLabel = this.isCouncilEnabled() ?
                                '{{ __('Select at least 2 models') }}' :
                                (this.selectedModels[0]?.label ?? '{{ __('None') }}');
                            return;
                        }

                        this.selectedModelLabel = this.selectedModels.length + ' {{ __('Models Selected') }}';
                    },
                    saveChanges() {
                        if (this.isCouncilEnabled() && this.selectedModels.length < 2) {
                            toastr.error('{{ __('Please select at least 2 models for Council Mode.') }}');
                            return;
                        }

                        this.setLocalStorage();
                        this.setCouncilModeStorage();
                        this.updateSelectionLabel();
                    },
                    onCouncilModeChange(enabled) {
                        this.councilMode = enabled;

                        this.normalizeSelectionByMode();
                        this.setCouncilModeStorage();
                        this.updateSelectionLabel();
                    },
                    init() {
                        const localStorageLastSelectedModels = this.getLocalStorage();
                        this.councilMode = this.getCouncilModeStorage();

                        if (this.councilMode && !this.canCouncilSelect) {
                            this.councilMode = false;
                        }

                        this.selectedModels = localStorageLastSelectedModels;
                        this.normalizeSelectionByMode();
                        // set local storage again on init to make sure we're in sync with available and active models
                        this.setLocalStorage();
                        this.setCouncilModeStorage();
                        this.updateSelectionLabel();

                        window.addEventListener('council-mode-changed', event => this.onCouncilModeChange(event.detail?.enabled ?? false));

                        document.addEventListener('chat-model-change', event => {
                            const {
                                model
                            } = event.detail;

                            if (!model || !model.trim()) return;

                            const selectEl = document.querySelector('#chatbot_front_model');
                            const matchedOption = Array.from(selectEl?.options || []).find(optionEl => optionEl?.value === model);

                            const modelObj = {
                                value: model,
                                label: matchedOption?.getAttribute('data-label') ?? model
                            };

                            // Only persist to localStorage if the model is in the active models list,
                            // otherwise just update the UI selection without saving (e.g. realtime model
                            // that isn't in the user's available models list)
                            const isInActiveModels = this.activeModels.some(m => (typeof m === 'string' ? m : m?.value) === model);

                            this.councilMode = false;
                            this.selectedModels = [modelObj];

                            if (isInActiveModels) {
                                this.setLocalStorage();
                            }

                            this.setCouncilModeStorage();
                            this.updateSelectionLabel();
                        });

                        Alpine.store('modelList', this);
                    }
                }))
            })
        })();
    </script>
@endpush
