<div
    class="flex w-full flex-col gap-6 lg:col-span-2"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'finance') }}"
>
    <div class="flex items-center justify-between">
        <h2 class="mb-0 font-bold">@lang('Finance')</h2>
        <x-button
            variant="link"
            href="{{ route('dashboard.admin.finance.plan.index') }}"
        >
            <span class="text-nowrap font-bold text-foreground"> {{ __('Visit Finance') }} </span>
            <x-tabler-chevron-right class="ms-auto size-4" />
        </x-button>
    </div>
    <x-card
        class:body="grid grid-cols-10 gap-9 px-2 sm:px-8 py-6"
        size="md"
    >
        <div class="col-span-10 flex flex-col gap-6 sm:col-span-8 sm:max-lg:col-span-7">
            <div class="flex items-end justify-between">
                <div class="flex flex-col gap-1">
                    <h4 class="flex items-center gap-1"> @lang('Total Earnings')
                        <x-info-tooltip text="{{ __('Your total income from all users and plans.') }}" />
                    </h4>
                    <p class="text-2xl font-semibold">
                        <span class="text-xl">{{ currency()->symbol }}</span>{{ cache()->get('total_sales') }}
                    </p>
                </div>
                <x-card
                    class="w-auto"
                    class:body="py-0 px-1 overflow-hidden"
                    size="sm"
                >
                    <ul class="flex flex-nowrap gap-1">
                        <li>
                            <span
                                class="finance-change-range-btn cursor-pointer px-2 py-0.5 text-center font-medium leading-7 text-foreground/80 [&.active]:rounded-md [&.active]:bg-[#EDEDED] [&.active]:text-foreground [&.active]:dark:bg-foreground/30"
                                onclick="updateDateRange(event,'day')"
                            >
                                {{ __('1D') }}
                            </span>
                        </li>
                        <li>
                            <span
                                class="finance-change-range-btn cursor-pointer px-2 py-0.5 text-center font-medium leading-7 text-foreground/80 [&.active]:rounded-md [&.active]:bg-[#EDEDED] [&.active]:text-foreground [&.active]:dark:bg-foreground/30"
                                onclick="updateDateRange(event, 'week')"
                            >
                                {{ __('1W') }}
                            </span>
                        </li>
                        <li>
                            <span
                                class="finance-change-range-btn active cursor-pointer px-2 py-0.5 text-center font-medium leading-7 text-foreground/80 [&.active]:rounded-md [&.active]:bg-[#EDEDED] [&.active]:text-foreground [&.active]:dark:bg-foreground/30"
                                onclick="updateDateRange(event, 'month')"
                            >
                                {{ __('1M') }}
                            </span>
                        </li>
                        <li>
                            <span
                                class="finance-change-range-btn cursor-pointer px-2 py-0.5 text-center font-medium leading-7 text-foreground/80 [&.active]:rounded-md [&.active]:bg-[#EDEDED] [&.active]:text-foreground [&.active]:dark:bg-foreground/30"
                                onclick="updateDateRange(event, 'year')"
                            >
                                {{ __('1Y') }}
                            </span>
                        </li>
                    </ul>
                </x-card>
            </div>
            <div
                class="min-h-[350px] w-full [&_.apexcharts-legend-text]:!m-0 [&_.apexcharts-legend-text]:!pe-2 [&_.apexcharts-legend-text]:ps-2 [&_.apexcharts-legend-text]:!text-foreground"
                id="chart-daily-sales"
            ></div>
        </div>
        <div class="col-span-10 grid grid-cols-1 content-center gap-7 sm:col-span-2 sm:max-lg:col-span-3">
            <x-card
                class:body="py-2"
                size="sm"
            >
                <div class="flex flex-col gap-3">
                    <span class="font-medium text-heading-foreground">{{ __('Referral Payouts') }}</span>
                    <h3 class="lqd-statistic-change m-0 flex items-center text-2xl sm:text-[30px]">
                        <span class="text-xl">{{ $currencySymbol }}</span>{{ cache()->get('referral_payout') }}
                    </h3>
                </div>
            </x-card>
            <x-card
                class:body="py-2"
                size="sm"
            >
                <div class="flex flex-col gap-3">
                    <span class="font-medium text-heading-foreground">{{ __('Total Spending') }}</span>
                    <h3 class="lqd-statistic-change m-0 flex items-center text-2xl sm:text-[30px]">
                        <span
                            class="text-xl">{{ $currencySymbol }}</span>{{ number_format(setting('total_spend', 0), 1) }}
                    </h3>
                </div>
            </x-card>
        </div>
    </x-card>
</div>
