<div class="flex flex-col gap-6">
    <div class="flex flex-col gap-2">
        <h3 class="text-lg font-semibold text-heading-foreground">
            @lang('Configure Shared Credits')
        </h3>
        <p class="text-xs text-foreground/70">
            @lang('Review and adjust the suggested shared credit amounts for each plan. The amounts are calculated based on the existing per-entity credits weighted by their shared credit index.')
        </p>
    </div>

    @foreach ($this->selectedPlans as $plan)
        @php
            $config = $planConfigs[$plan['id']] ?? null;
        @endphp

        @if ($config)
            <div
                class="flex flex-col gap-4 rounded-card border border-card-border bg-card-background p-5"
                wire:key="config-{{ $plan['id'] }}"
            >
                <div class="flex items-center justify-between">
                    <h4 class="text-sm font-semibold text-heading-foreground">
                        {{ $plan['name'] }}
                    </h4>
                    <span class="text-2xs text-foreground/60 capitalize">
                        {{ $plan['frequency'] }} &bull; ${{ number_format($plan['price'], 2) }}
                    </span>
                </div>

                <div class="flex flex-col gap-2">
                    <label
                        class="text-2xs font-medium text-heading-foreground"
                        for="credits-{{ $plan['id'] }}"
                    >
                        @lang('Shared Credits Amount')
                    </label>
                    <input
                        class="rounded-input border border-input-border bg-input-background px-3 py-2 text-sm text-input-foreground"
                        id="credits-{{ $plan['id'] }}"
                        type="number"
                        step="0.01"
                        min="0"
                        wire:model.lazy="planConfigs.{{ $plan['id'] }}.shared_credits_amount"
                    />
                </div>

                @if (! empty($config['breakdown']))
                    <details class="group">
                        <summary class="cursor-pointer text-2xs font-medium text-primary">
                            @lang('View credit breakdown') ({{ count($config['breakdown']) }} @lang('models'))
                        </summary>
                        <div class="mt-3 overflow-hidden rounded-lg border border-card-border">
                            <table class="w-full text-2xs">
                                <thead>
                                    <tr class="border-b border-card-border bg-foreground/5">
                                        <th class="px-3 py-2 text-start font-medium">@lang('Model')</th>
                                        <th class="px-3 py-2 text-start font-medium">@lang('Credits')</th>
                                        <th class="px-3 py-2 text-start font-medium">@lang('Index')</th>
                                        <th class="px-3 py-2 text-start font-medium">@lang('Shared Value')</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    @foreach ($config['breakdown'] as $item)
                                        <tr class="border-b border-card-border last:border-b-0">
                                            <td class="px-3 py-2">{{ $item['label'] }}</td>
                                            <td class="px-3 py-2">
                                                @if ($item['is_unlimited'])
                                                    <span class="text-green-500 font-medium">@lang('Unlimited')</span>
                                                @else
                                                    {{ $item['credit'] }}
                                                @endif
                                            </td>
                                            <td class="px-3 py-2">{{ $item['index'] }}</td>
                                            <td class="px-3 py-2">
                                                @if ($item['is_unlimited'])
                                                    &mdash;
                                                @else
                                                    {{ $item['shared_value'] }}
                                                @endif
                                            </td>
                                        </tr>
                                    @endforeach
                                </tbody>
                            </table>
                        </div>
                    </details>
                @endif

                @if (! empty($config['unlimited_models']))
                    <x-alert
                        class="text-2xs"
                        variant="warn"
                    >
                        @lang('This plan has :count unlimited model(s) that cannot be automatically converted:', ['count' => count($config['unlimited_models'])])
                        <strong>
                            {{ collect($config['unlimited_models'])->pluck('label')->implode(', ') }}
                        </strong>
                    </x-alert>
                @endif
            </div>
        @endif
    @endforeach
</div>
