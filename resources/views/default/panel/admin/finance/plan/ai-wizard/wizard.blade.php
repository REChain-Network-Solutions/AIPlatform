<div
    id="ai-plan-wizard-root"
    x-data="aiPlanWizard"
    x-effect="created && modal()?.modalOpen === false && window.location.reload()"
>
    <x-modal
        id="ai-plan-wizard-modal"
        type="page"
        class:modal="p-4 md:p-8"
        class:modal-content="w-[min(1200px,100%)] max-h-[92vh] rounded-card border border-card-border bg-background shadow-2xl"
        class:modal-head="w-full border-b border-solid border-card-border px-5 py-4 md:px-8"
        class:close-btn="relative top-0 end-0 lg:end-0 size-9 rounded-lg bg-transparent text-foreground hover:scale-100 hover:bg-foreground/10"
        class:modal-body="p-5 md:p-8"
    >
        <x-slot:title>
            <span class="inline-flex items-center gap-2">
                <x-tabler-sparkles class="size-5 text-primary" />
                {{ __('AI Plan Builder') }}
            </span>
        </x-slot:title>

        <x-slot:modal>
            <div class="grid w-full gap-8 lg:grid-cols-3">
                <div class="min-w-0 lg:col-span-2">
                    <div class="w-full">
                        <template x-if="step !== 'intro' && step !== 'done'">
                            <nav class="mb-8 flex flex-wrap items-center gap-2">
                                <template x-for="(item, index) in visibleSteps()" :key="item">
                                    <button
                                        type="button"
                                        class="inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium text-foreground/60 transition-colors [&.active]:bg-primary/10 [&.active]:text-primary [&.done]:text-foreground"
                                        :class="{ 'active': step === item, 'done': stepIndex(step) > index }"
                                        :disabled="stepIndex(step) < index"
                                        @click="stepIndex(step) > index && (step = item)"
                                    >
                                        <span
                                            class="grid size-5 place-items-center rounded-full bg-foreground/10 text-2xs"
                                            x-text="index + 1"
                                        ></span>
                                        <span x-text="stepLabels[item]"></span>
                                    </button>
                                </template>
                            </nav>
                        </template>

                        @include('panel.admin.finance.plan.ai-wizard.steps.intro')
                        @include('panel.admin.finance.plan.ai-wizard.steps.basics')
                        @include('panel.admin.finance.plan.ai-wizard.steps.credits')
                        @include('panel.admin.finance.plan.ai-wizard.steps.features')
                        @include('panel.admin.finance.plan.ai-wizard.steps.review')

                        <template x-if="step === 'done'">
                            <div class="flex flex-col items-center gap-4 py-16 text-center">
                                <span class="grid size-14 place-items-center rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                                    <x-tabler-check class="size-7" />
                                </span>
                                <h3
                                    class="m-0 text-lg font-semibold"
                                    x-text="created?.message"
                                ></h3>
                                <p class="m-0 max-w-md text-sm text-foreground/60">
                                    {{ __('Your plan is live with smart defaults. You can fine-tune every detail — per-model credits, templates and limits — in the advanced editor.') }}
                                </p>
                                <div class="mt-2 flex flex-wrap items-center justify-center gap-3">
                                    <x-button
                                        variant="primary"
                                        href="#"
                                        ::href="created?.editUrl"
                                    >
                                        <x-tabler-adjustments class="size-4" />
                                        {{ __('Review in Advanced Mode') }}
                                    </x-button>
                                    <x-button
                                        variant="outline"
                                        @click.prevent="window.location.reload()"
                                    >
                                        {{ __('Close') }}
                                    </x-button>
                                </div>
                            </div>
                        </template>

                        <template x-if="step !== 'intro' && step !== 'done'">
                            <div class="mt-8 flex items-center justify-between gap-3 border-t border-card-border pt-6">
                                <x-button
                                    variant="outline"
                                    @click.prevent="back()"
                                >
                                    {{ __('Back') }}
                                </x-button>

                                <div class="flex items-center gap-3">
                                    <template x-if="step !== 'review'">
                                        <x-button
                                            variant="primary"
                                            ::disabled="validating"
                                            @click.prevent="next()"
                                        >
                                            <span x-show="!validating">{{ __('Continue') }}</span>
                                            <span x-show="validating">{{ __('Checking...') }}</span>
                                        </x-button>
                                    </template>
                                    <template x-if="step === 'review'">
                                        <x-button
                                            variant="primary"
                                            ::disabled="submitting"
                                            @click.prevent="submitPlan()"
                                        >
                                            <x-tabler-sparkles class="size-4" />
                                            <span x-show="!submitting">{{ __('Create Plan') }}</span>
                                            <span x-show="submitting">{{ __('Creating...') }}</span>
                                        </x-button>
                                    </template>
                                </div>
                            </div>
                        </template>
                    </div>
                </div>

                @include('panel.admin.finance.plan.ai-wizard.copilot')
            </div>
        </x-slot:modal>
    </x-modal>

    @include('panel.admin.finance.plan.ai-wizard.script')
</div>
