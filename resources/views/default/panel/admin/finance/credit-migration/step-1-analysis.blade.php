<div class="flex flex-col gap-6">
    <div class="flex flex-col gap-2">
        <h3 class="text-lg font-semibold text-heading-foreground">
            @lang('Plan Analysis')
        </h3>
        <p class="text-xs text-foreground/70">
            @lang('Select the plans you want to migrate from separated credits to shared credits.')
        </p>
    </div>

    @error('selectedPlanIds')
        <x-alert variant="danger">
            {{ $message }}
        </x-alert>
    @enderror

    @if (count($plans) === 0)
        <x-alert variant="info">
            @lang('No plans available for migration. All plans are already using shared credits.')
        </x-alert>
    @else
        <div class="flex items-center gap-3">
            <x-button
                type="button"
                variant="ghost-shadow"
                size="sm"
                wire:click="selectAllPlans"
            >
                @lang('Select All')
            </x-button>
            <x-button
                type="button"
                variant="ghost-shadow"
                size="sm"
                wire:click="deselectAllPlans"
            >
                @lang('Deselect All')
            </x-button>
            <span class="text-2xs text-foreground/60">
                {{ count($selectedPlanIds) }} @lang('of') {{ count($plans) }} @lang('selected')
            </span>
        </div>

        <div class="overflow-hidden rounded-card border border-card-border">
            <table class="w-full text-xs">
                <thead>
                    <tr class="border-b border-card-border bg-card-background">
                        <th class="px-4 py-3 text-start font-medium text-foreground/70">
                        </th>
                        <th class="px-4 py-3 text-start font-medium text-foreground/70">
                            @lang('Plan Name')
                        </th>
                        <th class="px-4 py-3 text-start font-medium text-foreground/70">
                            @lang('Frequency')
                        </th>
                        <th class="px-4 py-3 text-start font-medium text-foreground/70">
                            @lang('Price')
                        </th>
                        <th class="px-4 py-3 text-start font-medium text-foreground/70">
                            @lang('Active Users')
                        </th>
                        <th class="px-4 py-3 text-start font-medium text-foreground/70">
                            @lang('Current Type')
                        </th>
                    </tr>
                </thead>
                <tbody>
                    @foreach ($plans as $plan)
                        <tr
                            class="cursor-pointer border-b border-card-border last:border-b-0 transition-colors hover:bg-foreground/5"
                            wire:click="togglePlan({{ $plan['id'] }})"
                            wire:key="plan-{{ $plan['id'] }}"
                        >
                            <td class="px-4 py-3">
                                <input
                                    type="checkbox"
                                    class="rounded border-input-border"
                                    @checked(in_array($plan['id'], $selectedPlanIds))
                                    wire:click.stop="togglePlan({{ $plan['id'] }})"
                                />
                            </td>
                            <td class="px-4 py-3 font-medium text-heading-foreground">
                                {{ $plan['name'] }}
                            </td>
                            <td class="px-4 py-3 capitalize">
                                {{ $plan['frequency'] }}
                            </td>
                            <td class="px-4 py-3">
                                ${{ number_format($plan['price'], 2) }}
                            </td>
                            <td class="px-4 py-3">
                                <span class="inline-flex items-center rounded-full bg-primary/10 px-2 py-0.5 text-2xs font-medium text-primary">
                                    {{ $plan['user_count'] }}
                                </span>
                            </td>
                            <td class="px-4 py-3 capitalize">
                                {{ $plan['credit_system_type'] }}
                            </td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </div>
    @endif
</div>
