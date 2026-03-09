<x-card
    class="flex flex-col"
    class:body="flex flex-col justify-center grow"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'cost-management') }}"
>
    <x-slot:head
        class="flex items-center justify-between px-5 py-3.5"
    >
        <div class="flex items-center gap-4">
            <x-lqd-icon class="bg-background text-heading-foreground dark:bg-foreground/5">
                <x-tabler-cash-banknote
                    class="size-6"
                    stroke-width="1.5"
                />
            </x-lqd-icon>
            <h4 class="m-0 flex items-center gap-1 text-base font-medium">
                {{ __('Cost Management') }}
                <x-info-tooltip text="{{ __('See how much each user costs vs. what they bring in.') }}" />
            </h4>
        </div>
    </x-slot:head>

    <x-card
        class="mt-4"
        class:body="flex items-center justify-between flex-nowrap py-4"
        size="sm"
    >
        <div class="flex grow flex-col items-start gap-px">
            <p class="mb-0 text-sm font-medium text-heading-foreground">
                @lang('Cost Per User')
            </p>
            <p class="mb-0 text-base font-semibold opacity-70 sm:text-[22px]">
                {{ currency()->symbol . (\App\Helpers\Classes\Helper::appIsDemo()  ? 7 : cache()->get('cost_per_user', 0.0)) }}
            </p>
            <p class="mb-0 text-2xs font-medium opacity-70">
                @lang('Average')
            </p>
        </div>
        <span class="h-28 w-px bg-border"></span>
        <div class="flex grow flex-col items-end gap-px">
            <p class="mb-0 text-sm font-medium text-heading-foreground">
                @lang('Income Per User')
            </p>
            <p class="mb-0 text-base font-semibold opacity-70 sm:text-[22px]">
                {{ currency()->symbol .(\App\Helpers\Classes\Helper::appIsDemo()  ? 5 : cache()->get('income_per_user', 0.0)) }}
            </p>
            <p class="mb-0 text-2xs font-medium opacity-70">
                @lang('Net Profit')
				{{ currency()->symbol . (\App\Helpers\Classes\Helper::appIsDemo() ? 2 : cache()->get('net_profit', 0.0)) }}
            </p>
        </div>
    </x-card>

    <div class="my-8 flex h-full flex-col justify-center gap-11">
        <div class="relative w-full">
            <ul class="flex h-2 w-full flex-nowrap items-center overflow-hidden rounded-7xl">
                <li class="h-full w-full bg-[#DC524C]"></li>
                <li class="h-full w-full bg-[#E77B35]"></li>
                <li class="h-full w-full bg-[#E0B43E]"></li>
                <li class="h-full w-full bg-[#20C69F]"></li>
            </ul>
            <div
                class="absolute top-1/2 flex -translate-y-1/2 bg-card-background px-2"
                style="left: {{ max(0, min(99, \App\Helpers\Classes\Helper::appIsDemo() ? 64 : cache()->get('net_profit', 0.0) )) }}%"
            >
                <span class="h-5 w-1.5 rounded-[55px] bg-foreground/30"></span>
            </div>
        </div>
        <ul class="flex flex-nowrap justify-between">
            <li class="flex items-center gap-2">
                <span class="h-[5px] w-4 rounded-[55px] bg-[#DC524C]"></span>
                <p class="mb-0 font-normal text-foreground">
                    {{ __('Very Poor') }}
                </p>
            </li>
            <li class="flex items-center gap-2">
                <span class="h-[5px] w-4 rounded-[55px] bg-[#E77B35]"></span>
                <p class="mb-0 font-normal text-foreground">
                    {{ __('Poor') }}
                </p>
            </li>
            <li class="flex items-center gap-2">
                <span class="h-[5px] w-4 rounded-[55px] bg-[#E0B43E]"></span>
                <p class="mb-0 font-normal text-foreground">
                    {{ __('Good') }}
                </p>
            </li>
            <li class="flex items-center gap-2">
                <span class="h-[5px] w-4 rounded-[55px] bg-[#20C69F]"></span>
                <p class="mb-0 font-normal text-foreground">
                    {{ __('Excellent') }}
                </p>
            </li>
        </ul>
    </div>

</x-card>
