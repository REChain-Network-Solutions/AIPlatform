<div class="flex flex-col gap-6">
    <div class="flex flex-col gap-2">
        <h3 class="text-lg font-semibold text-heading-foreground">
            @lang('Execute Migration')
        </h3>
        <p class="text-xs text-foreground/70">
            @if (! $executed)
                @lang('Review the migration details below and click Execute when ready. This action can be rolled back.')
            @else
                @lang('Migration has been executed successfully. Review the results below.')
            @endif
        </p>
    </div>

    @error('migration')
        <x-alert variant="danger">
            {{ $message }}
        </x-alert>
    @enderror

    @error('rollback')
        <x-alert variant="danger">
            {{ $message }}
        </x-alert>
    @enderror

    @if (! $executed)
        <div class="rounded-card border border-card-border bg-card-background p-5">
            <h4 class="mb-4 text-sm font-semibold text-heading-foreground">
                @lang('Pre-execution Preview')
            </h4>
            <div class="overflow-hidden rounded-lg border border-card-border">
                <table class="w-full text-xs">
                    <thead>
                        <tr class="border-b border-card-border bg-foreground/5">
                            <th class="px-4 py-2 text-start font-medium">@lang('Plan')</th>
                            <th class="px-4 py-2 text-start font-medium">@lang('Shared Credits')</th>
                            <th class="px-4 py-2 text-start font-medium">@lang('Active Users')</th>
                            <th class="px-4 py-2 text-start font-medium">@lang('Strategy')</th>
                        </tr>
                    </thead>
                    <tbody>
                        @foreach ($this->selectedPlans as $plan)
                            @php
                                $config = $planConfigs[$plan['id']] ?? null;
                            @endphp
                            <tr
                                class="border-b border-card-border last:border-b-0"
                                wire:key="preview-{{ $plan['id'] }}"
                            >
                                <td class="px-4 py-2 font-medium text-heading-foreground">
                                    {{ $plan['name'] }}
                                </td>
                                <td class="px-4 py-2">
                                    {{ number_format($config['shared_credits_amount'] ?? 0, 2) }}
                                </td>
                                <td class="px-4 py-2">
                                    {{ $plan['user_count'] }}
                                </td>
                                <td class="px-4 py-2 capitalize">
                                    {{ str_replace('_', ' ', $migrationStrategy) }}
                                </td>
                            </tr>
                        @endforeach
                    </tbody>
                </table>
            </div>

            <div class="mt-5">
                <x-button
                    class="w-full"
                    type="button"
                    variant="primary"
                    wire:click="executeMigration"
                    wire:loading.attr="disabled"
                    wire:target="executeMigration"
                    wire:confirm="{{ __('Are you sure you want to execute this migration? This will modify plan settings and user credits.') }}"
                >
                    <span wire:loading.remove wire:target="executeMigration">
                        @lang('Execute Migration')
                    </span>
                    <span wire:loading wire:target="executeMigration">
                        @lang('Executing...')
                    </span>
                </x-button>
            </div>
        </div>
    @else
        <x-alert variant="success">
            @lang('Migration completed successfully!')
        </x-alert>

        <div class="rounded-card border border-card-border bg-card-background p-5">
            <h4 class="mb-4 text-sm font-semibold text-heading-foreground">
                @lang('Results')
            </h4>
            <div class="flex flex-col gap-2 text-xs">
                <div class="flex items-center justify-between border-b border-card-border pb-2">
                    <span class="text-foreground/70">@lang('Plans migrated')</span>
                    <span class="font-semibold text-heading-foreground">{{ $results['plans_migrated'] ?? 0 }}</span>
                </div>
                <div class="flex items-center justify-between border-b border-card-border pb-2">
                    <span class="text-foreground/70">@lang('Users migrated')</span>
                    <span class="font-semibold text-heading-foreground">{{ $results['users_migrated'] ?? 0 }}</span>
                </div>
            </div>

            @if (! empty($results['plan_details']))
                <div class="mt-4 overflow-hidden rounded-lg border border-card-border">
                    <table class="w-full text-xs">
                        <thead>
                            <tr class="border-b border-card-border bg-foreground/5">
                                <th class="px-4 py-2 text-start font-medium">@lang('Plan')</th>
                                <th class="px-4 py-2 text-start font-medium">@lang('Shared Credits')</th>
                                <th class="px-4 py-2 text-start font-medium">@lang('Users Affected')</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach ($results['plan_details'] as $detail)
                                <tr class="border-b border-card-border last:border-b-0">
                                    <td class="px-4 py-2 font-medium text-heading-foreground">
                                        {{ $detail['name'] }}
                                    </td>
                                    <td class="px-4 py-2">
                                        {{ number_format($detail['shared_credits'], 2) }}
                                    </td>
                                    <td class="px-4 py-2">
                                        {{ $detail['users_affected'] }}
                                    </td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            @endif

            @if (! empty($results['errors']))
                <x-alert
                    class="mt-4"
                    variant="warn"
                >
                    <ul class="list-disc ps-4">
                        @foreach ($results['errors'] as $error)
                            <li>{{ $error }}</li>
                        @endforeach
                    </ul>
                </x-alert>
            @endif

            <div class="mt-5">
                <x-button
                    class="w-full"
                    type="button"
                    variant="danger"
                    wire:click="rollbackMigration"
                    wire:loading.attr="disabled"
                    wire:target="rollbackMigration"
                    wire:confirm="{{ __('Are you sure you want to rollback this migration? Plans and users will be reverted to their previous state.') }}"
                >
                    <span wire:loading.remove wire:target="rollbackMigration">
                        @lang('Rollback Migration')
                    </span>
                    <span wire:loading wire:target="rollbackMigration">
                        @lang('Rolling back...')
                    </span>
                </x-button>
            </div>
        </div>
    @endif
</div>
