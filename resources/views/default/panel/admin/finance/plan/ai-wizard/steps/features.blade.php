<template x-if="step === 'features'">
    <div>
        <x-form-step
            step="3"
            label="{{ __('Features') }}"
        />

        <div class="mt-6 flex flex-col gap-6">
            <div>
                <x-forms.input
                    id="ai-wizard-features"
                    type="textarea"
                    rows="3"
                    label="{{ __('Marketing Feature List') }}"
                    tooltip="{{ __('Comma-separated bullets shown on the pricing page. Ask the CoPilot to write them for you.') }}"
                    placeholder="{{ __('AI Writer, 100+ Templates, Priority Support') }}"
                    x-model="form.features"
                    ::class="errors.features && 'border-red-500'"
                />
                <template x-if="errors.features">
                    <p class="mt-1 text-xs text-red-500" x-text="errors.features[0]"></p>
                </template>
            </div>

            @if (count($planAiToolsMenu))
                <div x-show="form.type === 'subscription'">
                    <div class="mb-3 flex items-center justify-between gap-3">
                        <p class="m-0 text-sm font-medium">{{ __('AI Tools') }}</p>
                        <x-button
                            size="sm"
                            variant="ghost"
                            @click.prevent="toggleAll('plan_ai_tools')"
                        >
                            {{ __('Toggle all') }}
                        </x-button>
                    </div>
                    <div class="grid gap-x-6 gap-y-2 sm:grid-cols-2 lg:grid-cols-3">
                        @foreach ($planAiToolsMenu as $tool)
                            <x-forms.input
                                id="ai-wizard-tool-{{ $tool['key'] }}"
                                type="checkbox"
                                switcher
                                label="{{ __($tool['label'] ?? $tool['key']) }}"
                                x-model="form.plan_ai_tools['{{ $tool['key'] }}']"
                            />
                        @endforeach
                    </div>
                </div>
            @endif

            @if (count($planFeatureMenu))
                <div x-show="form.type === 'subscription'">
                    <div class="mb-3 flex items-center justify-between gap-3">
                        <p class="m-0 text-sm font-medium">{{ __('Platform Features') }}</p>
                        <x-button
                            size="sm"
                            variant="ghost"
                            @click.prevent="toggleAll('plan_features')"
                        >
                            {{ __('Toggle all') }}
                        </x-button>
                    </div>
                    <div class="grid gap-x-6 gap-y-2 sm:grid-cols-2 lg:grid-cols-3">
                        @foreach ($planFeatureMenu as $feature)
                            <x-forms.input
                                id="ai-wizard-feature-{{ $feature['key'] }}"
                                type="checkbox"
                                switcher
                                label="{{ __($feature['label'] ?? $feature['key']) }}"
                                x-model="form.plan_features['{{ $feature['key'] }}']"
                            />
                        @endforeach
                    </div>
                </div>
            @endif

            <x-alert variant="info">
                {{ __('Templates, per-tool limits and other advanced options are enabled with sensible defaults. Fine-tune them anytime in the classic plan editor.') }}
            </x-alert>
        </div>
    </div>
</template>
