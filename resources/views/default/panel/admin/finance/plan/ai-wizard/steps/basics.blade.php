<template x-if="step === 'basics'">
    <div>
        <x-form-step
            step="1"
            label="{{ __('Plan Basics') }}"
        />

        <div class="mt-6 flex flex-col gap-5">
            <div class="grid gap-5 sm:grid-cols-2">
                <div>
                    <x-forms.input
                        id="ai-wizard-type"
                        type="select"
                        label="{{ __('Plan Type') }}"
                        x-model="form.type"
                    >
                        <option value="subscription">{{ __('Subscription') }}</option>
                        <option value="prepaid">{{ __('Token Pack (one-time)') }}</option>
                    </x-forms.input>
                </div>
                <div>
                    <x-forms.input
                        id="ai-wizard-name"
                        label="{{ __('Plan Name') }}"
                        placeholder="{{ __('e.g. Pro') }}"
                        x-model="form.name"
                        ::class="errors.name && 'border-red-500'"
                    />
                    <template x-if="errors.name">
                        <p class="mt-1 text-xs text-red-500" x-text="errors.name[0]"></p>
                    </template>
                </div>
            </div>

            <div>
                <x-forms.input
                    id="ai-wizard-description"
                    type="textarea"
                    rows="3"
                    label="{{ __('Description') }}"
                    placeholder="{{ __('A short marketing description shown on the pricing page') }}"
                    x-model="form.description"
                    ::class="errors.description && 'border-red-500'"
                />
                <template x-if="errors.description">
                    <p class="mt-1 text-xs text-red-500" x-text="errors.description[0]"></p>
                </template>
            </div>

            <div class="grid gap-5 sm:grid-cols-3">
                <div>
                    <x-forms.input
                        id="ai-wizard-price"
                        type="number"
                        min="0"
                        step="0.01"
                        label="{{ __('Price') }} ({{ currency()->symbol ?? '$' }})"
                        x-model="form.price"
                        ::class="errors.price && 'border-red-500'"
                    />
                    <template x-if="errors.price">
                        <p class="mt-1 text-xs text-red-500" x-text="errors.price[0]"></p>
                    </template>
                </div>
                <div x-show="form.type === 'subscription'">
                    <x-forms.input
                        id="ai-wizard-frequency"
                        type="select"
                        label="{{ __('Billing Frequency') }}"
                        x-model="form.frequency"
                    >
                        @foreach (\App\Enums\Plan\FrequencyEnum::cases() as $frequency)
                            <option value="{{ $frequency->value }}">{{ $frequency->label() }}</option>
                        @endforeach
                    </x-forms.input>
                    <template x-if="errors.frequency">
                        <p class="mt-1 text-xs text-red-500" x-text="errors.frequency[0]"></p>
                    </template>
                </div>
                <div x-show="form.type === 'subscription'">
                    <x-forms.input
                        id="ai-wizard-trial-days"
                        type="number"
                        min="0"
                        step="1"
                        label="{{ __('Trial Days') }}"
                        tooltip="{{ __('0 disables the free trial.') }}"
                        x-model="form.trial_days"
                        ::class="errors.trial_days && 'border-red-500'"
                    />
                    <template x-if="errors.trial_days">
                        <p class="mt-1 text-xs text-red-500" x-text="errors.trial_days[0]"></p>
                    </template>
                </div>
            </div>

            <div class="flex flex-wrap items-center gap-8">
                <x-forms.input
                    id="ai-wizard-active"
                    type="checkbox"
                    switcher
                    label="{{ __('Active') }}"
                    x-model="form.active"
                />
                <x-forms.input
                    id="ai-wizard-featured"
                    type="checkbox"
                    switcher
                    label="{{ __('Highlight as Featured') }}"
                    x-model="form.is_featured"
                />
            </div>
        </div>
    </div>
</template>
