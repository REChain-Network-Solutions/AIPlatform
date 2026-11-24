<template x-if="step === 'review'">
    <div>
        <x-form-step
            step="4"
            label="{{ __('Review & Create') }}"
        />

        <div class="mt-6 flex flex-col gap-6">
            <div class="overflow-x-auto rounded-card border border-card-border">
                <table class="w-full text-sm">
                    <tbody class="divide-y divide-card-border">
                        <tr>
                            <td class="w-1/3 px-4 py-3 text-foreground/60">{{ __('Name') }}</td>
                            <td class="px-4 py-3 font-medium" x-text="form.name || '—'"></td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 text-foreground/60">{{ __('Type') }}</td>
                            <td
                                class="px-4 py-3"
                                x-text="form.type === 'prepaid' ? '{{ __('Token Pack') }}' : '{{ __('Subscription') }}'"
                            ></td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 text-foreground/60">{{ __('Price') }}</td>
                            <td class="px-4 py-3">
                                {{ currency()->symbol ?? '$' }}<span x-text="form.price"></span>
                                <span
                                    class="text-foreground/60"
                                    x-text="form.type === 'prepaid' ? ' · {{ __('one-time') }}' : ' / ' + form.frequency"
                                ></span>
                            </td>
                        </tr>
                        <tr x-show="form.type === 'subscription'">
                            <td class="px-4 py-3 text-foreground/60">{{ __('Trial') }}</td>
                            <td
                                class="px-4 py-3"
                                x-text="Number(form.trial_days) > 0 ? form.trial_days + ' {{ __('days') }}' : '{{ __('No trial') }}'"
                            ></td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 text-foreground/60">{{ __('Credits') }}</td>
                            <td class="px-4 py-3">
                                <span x-show="form.credit_system_type === 'shared'">
                                    <span x-text="form.shared_credits_amount"></span> {{ __('shared credits') }}
                                </span>
                                <span x-show="form.credit_system_type === 'separated'">
                                    {{ __('Per-model credits') }} · <span class="capitalize" x-text="form.credit_tier"></span> {{ __('tier') }}
                                    <span
                                        class="block text-xs text-foreground/60"
                                        x-show="creditLimitsSummary()"
                                        x-text="creditLimitsSummary()"
                                    ></span>
                                </span>
                            </td>
                        </tr>
                        <tr x-show="form.type === 'subscription'">
                            <td class="px-4 py-3 text-foreground/60">{{ __('AI Tools') }}</td>
                            <td class="px-4 py-3">
                                <span x-text="enabledCount('plan_ai_tools')"></span> {{ __('enabled') }}
                            </td>
                        </tr>
                        <tr x-show="form.type === 'subscription'">
                            <td class="px-4 py-3 text-foreground/60">{{ __('Platform Features') }}</td>
                            <td class="px-4 py-3">
                                <span x-text="enabledCount('plan_features')"></span> {{ __('enabled') }}
                            </td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 text-foreground/60">{{ __('Status') }}</td>
                            <td class="px-4 py-3">
                                <span x-text="form.active ? '{{ __('Active') }}' : '{{ __('Passive') }}'"></span>
                                <span x-show="form.is_featured"> · {{ __('Featured') }}</span>
                            </td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 text-foreground/60">{{ __('Feature List') }}</td>
                            <td class="px-4 py-3" x-text="form.features || '—'"></td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <x-alert variant="info">
                {{ __('Gateway products and prices will be created automatically for your enabled payment gateways.') }}
            </x-alert>

            <template x-if="errors.general">
                <x-alert variant="danger">
                    <span x-text="errors.general"></span>
                </x-alert>
            </template>
        </div>
    </div>
</template>
