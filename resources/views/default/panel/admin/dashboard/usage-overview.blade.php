<x-card
    class="lg:col-span-2"
    class:body="flex justify-between flex-wrap md:flex-nowrap py-6 px-10 max-sm:gap-8"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'usage-overview') }}"
>
    @php
        $sales_change = percentageChange($sales_prev_week, $sales_this_week);
        $users_change = percentageChange(cache()->get('last_week_new_users'), cache()->get('this_week_new_users'));
        $generated_change = percentageChange(cache()->get('usage_previous_week'), cache()->get('usage_this_week'));
        $dialy_activity_change = percentageChange(cache()->get('daily_activity_last_week'), cache()->get('daily_activity_this_week'));
    @endphp
    <div class="flex gap-4 max-sm:w-full">
        <div class="lqd-statistic-info flex grow flex-col gap-1">
            <div class="lqd-statistic-title mb-1 flex items-center gap-2 text-sm font-medium text-heading-foreground">
                <span class="size-2.5 rounded-sm bg-secondary"></span>
                {{ __('Earnings') }}
            </div>
            <h3 class="lqd-statistic-change mb-0.5 flex items-center gap-2 text-2xl sm:text-[30px]">
                @if (currencyShouldDisplayOnRight($currencySymbol))
                    {{ number_format(cache()->get('total_sales')) }} {{ $currencySymbol }}
                @else
                    {{ $currencySymbol }}{{ number_format(cache()->get('total_sales')) }}
                @endif
            </h3>
            <p class="mb-0 flex items-center gap-1 text-[12px] text-heading-foreground/50">
                @lang('vs Last Week') <x-change-indicator-plus-minus value="{{ \App\Helpers\Classes\Helper::appIsDemo() ? 34 : floatval($sales_change) }}" />
            </p>
        </div>
    </div>

    <span class="h-px w-full bg-border sm:h-auto sm:w-px"></span>

    <div class="flex gap-4 max-sm:w-full">
        <div class="lqd-statistic-info flex grow flex-col gap-1">
            <div class="lqd-statistic-title mb-1 flex items-center gap-2 text-sm font-medium text-heading-foreground">
                <span class="size-2.5 rounded-sm bg-secondary"></span>
                {{ __('New Users') }}
            </div>
            <h3 class="lqd-statistic-change mb-0.5 flex items-center gap-2 text-2xl sm:text-[30px]">
                {{ \App\Helpers\Classes\Helper::appIsDemo() ? 12 : cache()->get('this_week_new_users') }}
            </h3>
            <p class="mb-0 flex items-center gap-1 text-[12px] text-heading-foreground/50">
                @lang('vs Last Week') <x-change-indicator-plus-minus value="{{ \App\Helpers\Classes\Helper::appIsDemo() ? 18 : floatval($users_change) }}" />
            </p>
        </div>
    </div>

    <span class="h-px w-full bg-border sm:h-auto sm:w-px"></span>

    <div class="flex gap-4 max-sm:w-full">
        <div class="lqd-statistic-info flex grow flex-col gap-1">
            <div class="lqd-statistic-title mb-1 flex items-center gap-2 text-sm font-medium text-heading-foreground">
                <span class="size-2.5 rounded-sm bg-secondary"></span>
                {{ __('AI Usage') }}
            </div>
            <h3 class="lqd-statistic-change mb-0.5 flex items-center gap-2 text-2xl sm:text-[30px]">
                {{ cache()->get('total_usage') }}
            </h3>
            <p class="mb-0 flex items-center gap-1 text-[12px] text-heading-foreground/50">
                @lang('vs Last Week') <x-change-indicator-plus-minus value="{{ floatval($generated_change) }}" />
            </p>
        </div>
    </div>

    <span class="h-px w-full bg-border sm:h-auto sm:w-px"></span>

    <div class="flex gap-4 max-sm:w-full">
        <div class="lqd-statistic-info flex grow flex-col gap-1">
            <div class="lqd-statistic-title mb-1 flex items-center gap-2 text-sm font-medium text-heading-foreground">
                <span class="size-2.5 rounded-sm bg-secondary"></span>
                {{ __('Daily Visit') }}
            </div>
            <h3 class="lqd-statistic-change mb-0.5 flex items-center gap-2 text-2xl sm:text-[30px]">
                {{ \App\Helpers\Classes\Helper::appIsDemo() ? 2421 : cache()->get('total_daily_activity') }}
            </h3>
            <p class="mb-0 flex items-center gap-1 text-[12px] text-heading-foreground/50">
                @lang('vs Last Week')
				<x-change-indicator-plus-minus value="{{ \App\Helpers\Classes\Helper::appIsDemo() ? 14 : floatval($dialy_activity_change) }}" />
            </p>
        </div>
    </div>
</x-card>
