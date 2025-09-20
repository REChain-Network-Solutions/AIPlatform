<x-card
    class="flex flex-col"
    class:body="flex flex-col justify-center grow px-5 py-3.5"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'user-traffic') }}"
>
    <x-slot:head
        class="flex items-center justify-between"
    >
        <div class="flex items-center gap-4">
            <x-lqd-icon class="bg-background text-heading-foreground dark:bg-foreground/5">
                <x-tabler-clock-hour-3
                    class="size-6"
                    stroke-width="1.5"
                />
            </x-lqd-icon>
            <h4 class="m-0 flex items-center gap-1 text-base font-medium">
                {{ __('User Traffic') }}
                <x-info-tooltip text="{{ __('Where your users are coming from – search, social, direct, etc.') }}" />
            </h4>
        </div>
    </x-slot:head>

    <div class="flex items-center justify-between">
        <span class="text-base font-medium text-foreground">@lang('Pages')</span>
        <span class="text-base font-medium text-foreground">@lang('Views')</span>
    </div>

    <div
        class="min-h-[350px] w-full [&_.apexcharts-legend-text]:!m-0 [&_.apexcharts-legend-text]:!pe-2 [&_.apexcharts-legend-text]:ps-2 [&_.apexcharts-legend-text]:!text-foreground"
        id="user-traffic"
    ></div>
</x-card>
