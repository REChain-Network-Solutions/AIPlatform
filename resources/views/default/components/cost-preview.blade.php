@php
    $isSharedCreditUser = auth()->check() && auth()->user()->isSharedCreditUser()
        && app(\App\Services\SharedCredit\SharedCreditService::class)->isEnabled()
        && (bool) setting('shared_credit_show_cost_preview', false);
@endphp

@if ($isSharedCreditUser)
    <div
        {{ $attributes->withoutTwMergeClasses()->twMerge('lqd-cost-preview mt-3 flex justify-end') }}
        x-data="costPreview"
        x-cloak
        @generator-changed.window="onGeneratorChanged($event.detail)"
    >
        <div
            class="inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-3xs transition-all"
            :class="{
                'bg-foreground/[3%]': data && data.has_balance,
                'bg-amber-500/10': data && !data.has_balance && data.allow_negative,
                'bg-red-500/10': data && !data.has_balance && !data.allow_negative,
            }"
            x-show="visible"
            x-transition.opacity.duration.200ms
        >
            {{-- Loading --}}
            <template x-if="loading">
                <span class="flex items-center gap-1.5 text-foreground/50">
                    <x-tabler-loader-2 class="size-3.5 animate-spin" />
                    @lang('Calculating...')
                </span>
            </template>

            {{-- Error --}}
            <template x-if="!loading && error">
                <span class="text-red-500" x-text="error"></span>
            </template>

            {{-- Cost data --}}
            <template x-if="!loading && !error && data">
                <span class="flex items-center gap-2">
                    <span class="flex items-center gap-1 font-medium text-heading-foreground">
                        <x-tabler-bolt class="size-3.5 text-primary" />
                        <span x-text="Number(data.total_cost).toFixed(1)"></span>
                        @lang('credits')
                    </span>
                    <span class="size-1 rounded-full bg-foreground/20"></span>
                    <span class="text-foreground/50">
                        @lang('Balance:')
                        <span
                            class="font-medium"
                            :class="{
                                'text-foreground/70': data.has_balance,
                                'text-amber-500': !data.has_balance && data.allow_negative,
                                'text-red-500': !data.has_balance && !data.allow_negative,
                            }"
                            x-text="Number(data.balance).toFixed(1)"
                        ></span>
                    </span>
                    <template x-if="!data.has_balance && !data.allow_negative">
                        <span class="flex items-center gap-1 font-medium text-red-500">
                            <x-tabler-alert-circle class="size-3.5" />
                            @lang('Insufficient')
                        </span>
                    </template>
                    <template x-if="!data.has_balance && data.allow_negative">
                        <span class="flex items-center gap-1 font-medium text-amber-500">
                            <x-tabler-alert-triangle class="size-3.5" />
                            @lang('Low balance')
                        </span>
                    </template>
                </span>
            </template>
        </div>
    </div>

    @pushOnce('script')
        <script>
            document.addEventListener('alpine:init', () => {
                Alpine.data('costPreview', () => ({
                    loading: false,
                    error: null,
                    data: null,
                    visible: false,
                    _debounceTimer: null,
                    _lastGenerator: null,
                    _lastQuantity: null,
                    _lastAction: null,

                    onGeneratorChanged(detail) {
                        const generator = detail?.generator || detail;
                        const quantity = parseFloat(detail?.quantity) || 1;
                        const action = detail?.action || null;
                        const force = !!detail?._force;
                        this.debouncedFetch(generator, quantity, action, force);
                    },

                    debouncedFetch(generator, quantity, action = null, force = false) {
                        clearTimeout(this._debounceTimer);
                        this._debounceTimer = setTimeout(() => {
                            this.fetchPreview(generator, quantity, action, force);
                        }, 300);
                    },

                    async fetchPreview(generator, quantity = 1, action = null, force = false) {
                        if (!generator) {
                            this.visible = false;
                            return;
                        }

                        if (!force && generator === this._lastGenerator && quantity === this._lastQuantity && action === this._lastAction && this.data) {
                            return;
                        }

                        this._lastGenerator = generator;
                        this._lastQuantity = quantity;
                        this._lastAction = action;
                        this.loading = true;
                        this.error = null;
                        this.visible = true;

                        try {
                            const body = { generator, quantity };
                            if (action) body.action = action;

                            const response = await fetch('{{ route("shared-credit.cost-preview") }}', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                    'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]')?.content,
                                    'Accept': 'application/json',
                                },
                                body: JSON.stringify(body),
                            });

                            const result = await response.json();

                            if (!response.ok) {
                                this.error = result.message || '{{ __("Could not calculate cost.") }}';
                                return;
                            }

                            if (!result.is_shared_credit_user) {
                                this.visible = false;
                                return;
                            }

                            this.data = result.data;
                        } catch (e) {
                            this.error = '{{ __("Could not calculate cost.") }}';
                        } finally {
                            this.loading = false;
                        }
                    },
                }));
            });
        </script>
    @endPushOnce
@endif
