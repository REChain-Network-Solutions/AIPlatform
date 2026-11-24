<template x-if="step === 'credits'">
    <div>
        <x-form-step
            step="2"
            label="{{ __('Credits') }}"
        />

        <div class="mt-6 flex flex-col gap-6">
            <div class="grid gap-3 sm:grid-cols-2">
                <button
                    type="button"
                    class="rounded-card border border-card-border p-4 text-start transition-colors hover:border-primary [&.active]:border-primary [&.active]:bg-primary/5"
                    :class="{ 'active': form.credit_system_type === 'separated' }"
                    @click="form.credit_system_type = 'separated'"
                >
                    <span class="mb-1 block text-sm font-semibold">{{ __('Separated Credits') }}</span>
                    <span class="block text-xs text-foreground/60">
                        {{ __('Each AI model has its own credit pool. The AI distributes credits for you based on a tier.') }}
                    </span>
                </button>
                <button
                    type="button"
                    class="rounded-card border border-card-border p-4 text-start transition-colors hover:border-primary [&.active]:border-primary [&.active]:bg-primary/5"
                    :class="{ 'active': form.credit_system_type === 'shared' }"
                    @click="form.credit_system_type = 'shared'"
                >
                    <span class="mb-1 block text-sm font-semibold">{{ __('Shared Credits') }}</span>
                    <span class="block text-xs text-foreground/60">
                        {{ __('One credit pool shared across all models and tools. Simplest to explain to customers.') }}
                    </span>
                </button>
            </div>

            <div x-show="form.credit_system_type === 'shared'">
                <x-forms.input
                    id="ai-wizard-shared-credits"
                    type="number"
                    min="0"
                    step="1"
                    label="{{ __('Shared Credits Amount') }}"
                    tooltip="{{ __('Total credits a subscriber receives each cycle.') }}"
                    x-model="form.shared_credits_amount"
                    ::class="errors.shared_credits_amount && 'border-red-500'"
                />
                <template x-if="errors.shared_credits_amount">
                    <p class="mt-1 text-xs text-red-500" x-text="errors.shared_credits_amount[0]"></p>
                </template>
            </div>

            <div x-show="form.credit_system_type === 'separated'">
                <p class="mb-3 text-sm font-medium">
                    {{ __('Credit Tier') }}
                </p>
                <div class="grid gap-3 sm:grid-cols-4">
                    <template x-for="tier in creditTiers" :key="tier.key">
                        <button
                            type="button"
                            class="rounded-card border border-card-border p-3 text-center transition-colors hover:border-primary [&.active]:border-primary [&.active]:bg-primary/5"
                            :class="{ 'active': form.credit_tier === tier.key }"
                            @click="form.credit_tier = tier.key"
                        >
                            <span class="block text-sm font-semibold" x-text="tier.label"></span>
                            <span class="block text-xs text-foreground/60" x-text="tier.multiplier + '× {{ __('default credits') }}'"></span>
                        </button>
                    </template>
                </div>
                <p class="mt-3 text-xs text-foreground/60">
                    {{ __('The tier scales the default per-model credit limits. You can fine-tune individual models later in the classic plan editor.') }}
                </p>
                <template x-if="errors.credit_tier">
                    <p class="mt-1 text-xs text-red-500" x-text="errors.credit_tier[0]"></p>
                </template>

                <p class="mb-3 mt-6 text-sm font-medium">
                    {{ __('Per-Category Credit Overrides') }}
                </p>
                <div class="grid gap-3 sm:grid-cols-3">
                    <x-forms.input
                        id="ai-wizard-credit-limit-word"
                        type="number"
                        min="0"
                        step="1"
                        label="{{ __('Word Models') }}"
                        placeholder="{{ __('Auto (tier)') }}"
                        x-model="form.credit_limits.word"
                    />
                    <x-forms.input
                        id="ai-wizard-credit-limit-image"
                        type="number"
                        min="0"
                        step="1"
                        label="{{ __('Image Models') }}"
                        placeholder="{{ __('Auto (tier)') }}"
                        x-model="form.credit_limits.image"
                    />
                    <x-forms.input
                        id="ai-wizard-credit-limit-video"
                        type="number"
                        min="0"
                        step="1"
                        label="{{ __('Video Models') }}"
                        placeholder="{{ __('Auto (tier)') }}"
                        x-model="form.credit_limits.video"
                    />
                    <x-forms.input
                        id="ai-wizard-credit-limit-audio"
                        type="number"
                        min="0"
                        step="1"
                        label="{{ __('Audio Models') }}"
                        placeholder="{{ __('Auto (tier)') }}"
                        x-model="form.credit_limits.audio"
                    />
                    <x-forms.input
                        id="ai-wizard-credit-limit-presentation"
                        type="number"
                        min="0"
                        step="1"
                        label="{{ __('Presentation Models') }}"
                        placeholder="{{ __('Auto (tier)') }}"
                        x-model="form.credit_limits.presentation"
                    />
                </div>
                <p class="mt-3 text-xs text-foreground/60">
                    {{ __('Leave a field empty to keep the tier default for that category.') }}
                </p>
                <template x-if="Object.keys(errors).some((key) => key.startsWith('credit_limits'))">
                    <p
                        class="mt-1 text-xs text-red-500"
                        x-text="errors[Object.keys(errors).find((key) => key.startsWith('credit_limits'))][0]"
                    ></p>
                </template>
            </div>
        </div>
    </div>
</template>
