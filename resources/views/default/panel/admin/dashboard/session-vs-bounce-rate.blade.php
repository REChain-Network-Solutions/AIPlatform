<x-card
    class="flex flex-col"
    class:body="flex flex-col justify-start grow py-8"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'sessions-vs-bounce-rate') }}"
>
    <x-slot:head
        class="flex items-center justify-between px-5 py-3.5"
    >
        <div class="flex items-center gap-4">
            <x-lqd-icon class="bg-background text-heading-foreground dark:bg-foreground/5">
                <x-tabler-clock-hour-4
                    class="size-6"
                    stroke-width="1.5"
                />
            </x-lqd-icon>
            <h4 class="m-0 flex items-center gap-1 text-base font-medium">
                {{ __('Session vs Bounce Rate') }}
                <x-tabler-info-circle class="size-6 fill-[#41444A4D] stroke-background dark:fill-foreground"></x-tabler-info-circle>
            </h4>
        </div>
    </x-slot:head>
    <div class="flex h-full flex-col gap-7">
        <x-card
            class="w-full"
            class:body="flex flex-col justify-center items-center gap-2"
        >
            <h1 class="mb-0 text-center text-4xl"><span class="text-foreground/70">00:</span>08:30</h1>
            <span class="text-center text-lg">@lang('Avarage User Session')</span>
        </x-card>
        <ul class="flex flex-col gap-4">
            <li class="flex items-center justify-between border-b pb-4">
                <div class="flex items-center gap-2.5 font-medium text-foreground">
                    <x-tabler-bounce-left-filled
                        class="size-8"
                        stroke-width="1.5"
                    />
                    @lang('Bounce Rate')
                </div>
                <span class="text-sm font-medium leading-5 text-foreground/50">6%</span>
            </li>
            <li class="flex items-center justify-between border-b pb-4">
                <div class="flex items-center gap-2.5 font-medium text-foreground">
                    <x-tabler-hourglass
                        class="size-8"
                        stroke-width="1.5"
                    />
                    @lang('Total Sessions')
                </div>
                <span class="text-sm font-medium leading-5 text-foreground/50">11.5k</span>
            </li>
            <li class="flex items-center justify-between border-b pb-4">
                <div class="flex items-center gap-2.5 font-medium text-foreground">
                    <x-tabler-eye
                        class="size-8"
                        stroke-width="1.5"
                    />
                    @lang('Averate Views Per Session')
                </div>
                <span class="text-sm font-medium leading-5 text-foreground/50">234</span>
            </li>
        </ul>
    </div>
</x-card>
