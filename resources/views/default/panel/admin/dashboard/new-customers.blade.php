<x-card
    class="flex flex-col"
    class:body="flex flex-col justify-center grow"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'new-customers') }}"
>
    <x-slot:head
        class="flex items-center justify-between px-5 py-3.5"
    >
        <div class="flex items-center gap-4">
            <x-lqd-icon class="bg-background text-heading-foreground dark:bg-foreground/5">
                <x-tabler-device-analytics
                    class="size-6"
                    stroke-width="1.5"
                />
            </x-lqd-icon>
            <h4 class="m-0 flex items-center gap-1 text-base font-medium">
                {{ __('New Customers') }}
                <x-info-tooltip text="{{ __('Number of new signups over a selected period.') }}" />
            </h4>
        </div>
        <div class="flex items-center gap-5">
            <div class="flex items-center gap-2.5">
                <span class="size-2.5 rounded-sm bg-primary"></span>
                <p class="mb-0 text-base font-medium text-heading-foreground">@lang('Paid')</p>
            </div>
            <div class="flex items-center gap-2.5">
                <span class="size-2.5 rounded-sm bg-secondary"></span>
                <p class="mb-0 text-base font-medium text-heading-foreground">@lang('Free')</p>
            </div>
        </div>
    </x-slot:head>

    <div
        class="min-h-[350px] w-full [&_.apexcharts-legend-text]:!m-0 [&_.apexcharts-legend-text]:!pe-2 [&_.apexcharts-legend-text]:ps-2 [&_.apexcharts-legend-text]:!text-foreground"
        id="new-customers"
    ></div>
</x-card>
