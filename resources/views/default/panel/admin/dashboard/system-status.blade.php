<x-card
    class="flex flex-col"
    class:body="flex flex-col justify-center items-center grow"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'system-status') }}"
>
    <x-slot:head
        class="flex items-center justify-between px-5 py-3.5"
    >
        <div class="flex items-center gap-4">
            <x-lqd-icon class="bg-background text-heading-foreground dark:bg-foreground/5">
                <x-tabler-device-desktop-heart
                    class="size-6"
                    stroke-width="1.5"
                />
            </x-lqd-icon>
            <h4 class="m-0 flex items-center gap-1 text-base font-medium">
                {{ __('System Status') }}
                <x-info-tooltip text="{{ __('Server disk space.') }}" />
            </h4>
        </div>
        <x-button
            variant="link"
            href="{{ route('dashboard.admin.health.index') }}"
        >
            <span class="text-nowrap font-bold text-foreground"> {{ __('View Logs') }} </span>
            <x-tabler-chevron-right class="size-4 rtl:rotate-180" />
        </x-button>
    </x-slot:head>

    <div
        class="min-h-[350px] w-full [&_.apexcharts-legend-text]:!m-0 [&_.apexcharts-legend-text]:!pe-2 [&_.apexcharts-legend-text]:ps-2 [&_.apexcharts-legend-text]:!text-foreground"
        id="system-status"
    ></div>

    <div class="absolute inset-0 flex h-full flex-col justify-center">
        <h1
            class="text-center text-3xl font-bold leading-10 text-foreground/90 max-sm:mt-6 sm:text-[52px] sm:leading-[78px] md:max-lg:mt-6 md:max-lg:text-[40px] md:max-lg:leading-10">
            {{ cache()->get('available_diskspace') }}%</h1>
        <span class="text-center text-[15px] text-foreground/50">{{ __('Available Disk Space') }}</span>
    </div>
</x-card>
