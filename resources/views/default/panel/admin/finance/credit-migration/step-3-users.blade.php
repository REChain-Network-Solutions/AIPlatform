<div class="flex flex-col gap-6">
    <div class="flex flex-col gap-2">
        <h3 class="text-lg font-semibold text-heading-foreground">
            @lang('User Migration Strategy')
        </h3>
        <p class="text-xs text-foreground/70">
            @lang('Choose how existing users on these plans should be migrated to shared credits.')
        </p>
    </div>

    <div class="flex flex-col gap-3">
        <label
            class="flex cursor-pointer items-start gap-3 rounded-card border border-card-border p-4 transition-colors has-[:checked]:border-primary has-[:checked]:bg-primary/5"
        >
            <input
                class="mt-0.5"
                type="radio"
                wire:model="migrationStrategy"
                value="immediate"
            />
            <div class="flex flex-col gap-1">
                <span class="text-sm font-semibold text-heading-foreground">
                    @lang('Immediate Migration')
                </span>
                <span class="text-2xs text-foreground/70">
                    @lang('Migrate all active users now. Their shared credit balance will be calculated based on the remaining time in their current billing cycle (pro-rata).')
                </span>
            </div>
        </label>

        <label
            class="flex cursor-pointer items-start gap-3 rounded-card border border-card-border p-4 transition-colors has-[:checked]:border-primary has-[:checked]:bg-primary/5"
        >
            <input
                class="mt-0.5"
                type="radio"
                wire:model="migrationStrategy"
                value="on_renewal"
            />
            <div class="flex flex-col gap-1">
                <span class="text-sm font-semibold text-heading-foreground">
                    @lang('On Renewal')
                </span>
                <span class="text-2xs text-foreground/70">
                    @lang('Only convert the plans now. Users will receive shared credits when their subscription renews. Existing per-entity credits remain active until renewal.')
                </span>
            </div>
        </label>
    </div>

    <div class="rounded-card border border-card-border bg-card-background p-5">
        <h4 class="mb-4 text-sm font-semibold text-heading-foreground">
            @lang('Migration Summary')
        </h4>
        <div class="flex flex-col gap-2 text-xs">
            <div class="flex items-center justify-between border-b border-card-border pb-2">
                <span class="text-foreground/70">@lang('Plans to migrate')</span>
                <span class="font-medium text-heading-foreground">{{ count($selectedPlanIds) }}</span>
            </div>
            <div class="flex items-center justify-between border-b border-card-border pb-2">
                <span class="text-foreground/70">@lang('Active users affected')</span>
                <span class="font-medium text-heading-foreground">{{ $this->activeUserCount }}</span>
            </div>
            <div class="flex items-center justify-between border-b border-card-border pb-2">
                <span class="text-foreground/70">@lang('Strategy')</span>
                <span class="font-medium capitalize text-heading-foreground">
                    {{ str_replace('_', ' ', $migrationStrategy) }}
                </span>
            </div>
            @foreach ($this->selectedPlans as $plan)
                @php
                    $config = $planConfigs[$plan['id']] ?? null;
                @endphp
                <div
                    class="flex items-center justify-between"
                    wire:key="summary-{{ $plan['id'] }}"
                >
                    <span class="text-foreground/70">{{ $plan['name'] }}</span>
                    <span class="font-medium text-heading-foreground">
                        {{ number_format($config['shared_credits_amount'] ?? 0, 2) }} @lang('credits')
                    </span>
                </div>
            @endforeach
        </div>
    </div>
</div>
