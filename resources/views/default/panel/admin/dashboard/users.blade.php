@php
    $online_user_percentage = round((cache()->get('online_user', 0) / cache()->get('total_user', 1)) * 100, 0);
@endphp
<x-card
    class="flex flex-col"
    class:body="flex flex-col justify-center grow"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'users') }}"
>
    <x-slot:head
        class="flex items-center justify-between px-5 py-3.5"
    >
        <div class="flex items-center gap-4">
            <x-lqd-icon class="bg-background text-heading-foreground dark:bg-foreground/5">
                <x-tabler-user-circle
                    class="size-6"
                    stroke-width="1.5"
                />
            </x-lqd-icon>
            <h4 class="m-0 flex items-center gap-1 text-base font-medium">
                {{ __('Users') }}
                <x-info-tooltip text="{{ __('Overview of current user types and status.') }}" />
            </h4>
        </div>
        <x-button
            variant="link"
            href="{{ route('dashboard.admin.users.dashboard') }}"
        >
            <span class="text-nowrap font-bold"> {{ __('Manage Users') }} </span>
            <x-tabler-chevron-right class="size-4 rtl:rotate-180" />
        </x-button>
    </x-slot:head>
    <div class="flex flex-col gap-4">
        <div class="flex flex-col gap-4">
            <div class="flex justify-between">
                <span class="text-base font-medium"><strong class="me-2 text-xl font-bold">{{ \App\Helpers\Classes\Helper::appIsDemo() ? 23 : cache()->get('online_user') }}</strong>Online
                    Users</span>
                <span class="text-xs font-bold">{{ $online_user_percentage }}%</span>
            </div>
            <div class="flex w-full rounded-7xl border p-4">
                <div class="relative h-2.5 w-full rounded-7xl bg-foreground/10">
                    <span
                        class="absolute left-0 top-0 h-2.5 rounded-7xl bg-[#1FBA96]"
                        style="width: {{ $online_user_percentage }}%"
                    ></span>
                </div>
            </div>
        </div>
        <ul class="flex flex-col">
            <li class="flex items-center justify-between border-b border-card-border py-2.5">
                <div class="flex items-center gap-2.5">
                    <span class="size-2.5 rounded-sm bg-[#818B99]"></span>
                    <p class="mb-0 text-base font-medium">{{ __('Total Users') }}</p>
                </div>
                <x-money-with-unit :value="cache()->get('total_user', 0)" />
            </li>
            <li class="flex items-center justify-between border-b border-card-border py-2.5">
                <div class="flex items-center gap-2.5">
                    <span class="size-2.5 rounded-sm bg-[#20C69F]"></span>
                    <p class="mb-0 text-base font-medium">{{ __('Free Users') }}</p>
                </div>
                <x-money-with-unit :value="cache()->get('free_user', 0)" />
            </li>
            <li class="flex items-center justify-between border-b border-card-border py-2.5">
                <div class="flex items-center gap-2.5">
                    <span class="size-2.5 rounded-sm bg-primary"></span>
                    <p class="mb-0 text-base font-medium">{{ __('Paid Users') }}</p>
                </div>
                <x-money-with-unit :value="cache()->get('paid_user', 0)" />
            </li>
            <li class="flex items-center justify-between py-2.5">
                <div class="flex items-center gap-2.5">
                    <span class="size-2.5 rounded-sm bg-[#93C5FD]"></span>
                    <p class="mb-0 text-base font-medium">{{ __('Trial Users') }}</p>
                </div>
                <x-money-with-unit :value="cache()->get('trial_user', 0)" />
            </li>
        </ul>
    </div>
</x-card>
