<template x-if="step === 'intro'">
    <div>
        <h3 class="mb-1 text-lg font-semibold">
            {{ __('Let AI help you build a new plan') }}
        </h3>
        <p class="mb-6 text-sm text-foreground/60">
            {{ __('Here is a quick look at your current line-up. Pick an AI suggestion to prefill the wizard, or start from scratch.') }}
        </p>

        <div class="mb-8">
            <h4 class="mb-3 text-2xs font-medium uppercase tracking-wide text-foreground/50">
                {{ __('Your current plans') }}
            </h4>

            @if ($plans->isEmpty())
                <x-alert variant="info">
                    {{ __('You have no plans yet. Your first plan defines your pricing — the CoPilot can help you position it.') }}
                </x-alert>
            @else
                <div class="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                    @foreach ($plans->sortBy('price') as $existingPlan)
                        <div class="rounded-card border border-card-border bg-surface p-4">
                            <div class="mb-1 flex items-center justify-between gap-2">
                                <span class="truncate text-sm font-medium">{{ $existingPlan->name }}</span>
                                <x-badge
                                    class="text-2xs"
                                    variant="{{ $existingPlan->active ? 'success' : 'danger' }}"
                                >
                                    {{ $existingPlan->active ? __('Active') : __('Passive') }}
                                </x-badge>
                            </div>
                            <p class="m-0 text-xs text-foreground/60">
                                {{ currency()->symbol ?? '$' }}{{ $existingPlan->price }}
                                &middot;
                                {{ $existingPlan->type === 'prepaid' ? __('Token Pack') : __(formatCamelCase($existingPlan->frequency)) }}
                            </p>
                        </div>
                    @endforeach
                </div>
            @endif
        </div>

        <div class="mb-8">
            <h4 class="mb-3 text-2xs font-medium uppercase tracking-wide text-foreground/50">
                {{ __('AI suggestions') }}
            </h4>

            <template x-if="!presetsLoaded && !loadingPresets">
                <div class="flex flex-col items-start gap-2">
                    <x-button
                        variant="outline"
                        @click.prevent="loadPresets()"
                    >
                        <x-tabler-sparkles class="size-4 text-primary" />
                        {{ __('Generate suggestions with AI') }}
                    </x-button>
                    <p class="m-0 text-xs text-foreground/60">
                        {{ __('AI analyzes your existing plans and proposes new tiers to fill the gaps.') }}
                    </p>
                </div>
            </template>

            <template x-if="loadingPresets">
                <div class="grid gap-3 sm:grid-cols-3">
                    <template x-for="i in 3" :key="i">
                        <div class="h-32 animate-pulse rounded-card border border-card-border bg-foreground/5"></div>
                    </template>
                </div>
            </template>

            <template x-if="!loadingPresets && presets.length">
                <div class="grid gap-3 sm:grid-cols-3">
                    <template x-for="(preset, index) in presets" :key="index">
                        <button
                            type="button"
                            class="group flex flex-col items-start gap-1 rounded-card border border-card-border bg-card-background p-4 text-start transition-colors hover:border-primary"
                            @click="applyPreset(preset)"
                        >
                            <span class="inline-flex items-center gap-1.5 text-sm font-semibold">
                                <x-tabler-sparkles class="size-4 text-primary" />
                                <span x-text="preset.name"></span>
                            </span>
                            <span class="text-xs font-medium text-primary">
                                {{ currency()->symbol ?? '$' }}<span x-text="preset.price"></span>
                                <span x-text="preset.type === 'prepaid' ? '{{ __('one-time') }}' : '/ ' + preset.frequency"></span>
                            </span>
                            <span
                                class="text-xs text-foreground/60"
                                x-text="preset.reason"
                            ></span>
                        </button>
                    </template>
                </div>
            </template>

            <template x-if="presetsLoaded && !loadingPresets && !presets.length">
                <p class="m-0 text-sm text-foreground/60">
                    {{ __('AI suggestions are unavailable right now. You can still start from scratch or ask the CoPilot.') }}
                </p>
            </template>
        </div>

        <x-button
            variant="outline"
            @click.prevent="startFromScratch()"
        >
            {{ __('Start from scratch') }}
        </x-button>
    </div>
</template>
