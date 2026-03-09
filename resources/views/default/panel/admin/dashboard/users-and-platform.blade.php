<div
    class="flex w-full flex-col gap-11 lg:col-span-2"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'users-and-platform') }}"
>
    <div class="flex items-center justify-between">
        <h2 class="mb-0 font-bold">@lang('Users and Platform')</h2>
        <x-button
            variant="link"
            href="{{ route('dashboard.user.index') }}"
        >
            <span class="text-nowrap font-bold text-foreground"> {{ __('Visit User Dashboard') }} </span>
            <x-tabler-chevron-right class="size-4 rtl:rotate-180" />
        </x-button>
    </div>
    <x-card class:body="flex justify-between flex-wrap md:flex-nowrap py-6 px-10 max-sm:gap-8">
        @php
            $users_change = percentageChange(cache()->get('last_week_new_users'), cache()->get('this_week_new_users'));
            $subscribers_change = percentageChange(cache()->get('last_week_subscribers'), cache()->get('this_week_subscribers'));
            $referrals_change = percentageChange(cache()->get('last_week_referrals'), cache()->get('this_week_referrals'));
            $total_users_change = percentageChange(cache()->get('last_week_total_users'), cache()->get('this_week_total_users'));
        @endphp
        <div class="flex gap-4 max-sm:w-full">
            <div class="lqd-statistic-info flex grow flex-col gap-1">
                <div
                    class="lqd-statistic-title mb-1 flex items-center gap-2 text-sm font-medium text-heading-foreground">
                    <span class="size-2.5 rounded-sm bg-[#93C5FD]"></span>
                    {{ __('New Users') }}
                </div>
                <h3 class="lqd-statistic-change m-0.5 flex items-center gap-2 text-2xl sm:text-[30px]">
                    {{ cache()->get('this_week_new_users') }}
                </h3>
                <p class="mb-0 flex items-center gap-1 text-[12px] text-heading-foreground/50">
                    @lang('vs Last week') <x-change-indicator-plus-minus value="{{ floatval($users_change) }}" />
                </p>
            </div>
        </div>

        <span class="h-px w-full bg-border sm:h-auto sm:w-px"></span>

        <div class="flex gap-4 max-sm:w-full">
            <div class="lqd-statistic-info flex grow flex-col gap-1">
                <div
                    class="lqd-statistic-title mb-1 flex items-center gap-2 text-sm font-medium text-heading-foreground">
                    <span class="size-2.5 rounded-sm bg-secondary"></span>
                    {{ __('New Subscribers') }}
                </div>
                <h3 class="lqd-statistic-change m-0.5 flex items-center gap-2 text-2xl sm:text-[30px]">
                    {{ cache()->get('this_week_subscribers') }}
                </h3>
                <p class="mb-0 flex items-center gap-1 text-[12px] text-heading-foreground/50">
                    @lang('vs Last week') <x-change-indicator-plus-minus value="{{ floatval($subscribers_change) }}" />
                </p>
            </div>
        </div>

        <span class="h-px w-full bg-border sm:h-auto sm:w-px"></span>

        <div class="flex gap-4 max-sm:w-full">
            <div class="lqd-statistic-info flex grow flex-col gap-1">
                <div
                    class="lqd-statistic-title mb-1 flex items-center gap-2 text-sm font-medium text-heading-foreground">
                    <span class="size-2.5 rounded-sm bg-[#89E1C5]"></span>
                    {{ __('New Referrals') }}
                </div>
                <h3 class="lqd-statistic-change m-0.5 flex items-center gap-2 text-2xl sm:text-[30px]">
                    {{ cache()->get('this_week_referrals') }}
                </h3>
                <p class="mb-0 flex items-center gap-1 text-[12px] text-heading-foreground/50">
                    @lang('vs Last week') <x-change-indicator-plus-minus value="{{ floatval($referrals_change) }}" />
                </p>
            </div>
        </div>

        <span class="h-px w-full bg-border sm:h-auto sm:w-px"></span>

        <div class="flex gap-4 max-sm:w-full">
            <div class="lqd-statistic-info flex grow flex-col gap-1">
                <div
                    class="lqd-statistic-title mb-1 flex items-center gap-2 text-sm font-medium text-heading-foreground">
                    <span class="size-2.5 rounded-sm bg-[#93C5FD]"></span>
                    {{ __('Total Users') }}
                </div>
                <h3 class="lqd-statistic-change m-0.5 flex items-center gap-2 text-2xl sm:text-[30px]">
                    {{ \App\Helpers\Classes\Helper::appIsDemo() ? 2 : cache()->get('this_week_total_users') }}
                </h3>
                <p class="mb-0 flex items-center gap-1 text-[12px] text-heading-foreground/50">
                    @lang('vs Last week') <x-change-indicator-plus-minus value="{{ \App\Helpers\Classes\Helper::appIsDemo() ? 10 : floatval($total_users_change) }}" />
                </p>
            </div>
        </div>
    </x-card>
</div>
