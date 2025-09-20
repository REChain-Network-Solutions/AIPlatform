<x-card
    class="flex flex-col"
    class:body="flex flex-col justify-center grow"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'top-countries') }}"
>
    <x-slot:head
        class="flex items-center justify-between px-5 py-3.5"
    >
        <div class="flex items-center gap-4">
            <x-lqd-icon class="bg-background text-heading-foreground dark:bg-foreground/5">
                <x-tabler-globe
                    class="size-6"
                    stroke-width="1.5"
                />
            </x-lqd-icon>
            <h4 class="m-0 flex items-center gap-1 text-base font-medium">
                {{ __('Top Countries') }}
                <x-info-tooltip text="{{ __('Countries with the most users.') }}" />
            </h4>
        </div>
    </x-slot:head>

    <div class="flex items-center justify-between">
        <span class="text-base font-medium text-foreground">@lang('Countries')</span>
        <span class="text-base font-medium text-foreground">@lang('Users')</span>
    </div>

    <div
        class="min-h-[350px] w-full [&_.apexcharts-legend-text]:!m-0 [&_.apexcharts-legend-text]:!pe-2 [&_.apexcharts-legend-text]:ps-2 [&_.apexcharts-legend-text]:!text-foreground"
        id="top-countries"
    ></div>
</x-card>
