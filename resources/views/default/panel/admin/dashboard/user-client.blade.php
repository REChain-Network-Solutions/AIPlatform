<x-card
    class="flex flex-col"
    class:body="flex flex-col justify-center grow min-h-[350px] w-full"
    id="{{ 'admin-card-' . ($widget?->name?->value ?? 'user-client') }}"
>
    <x-slot:head
        class="flex items-center justify-between px-5 py-3.5"
    >
        <div class="flex items-center gap-4">
            <x-lqd-icon class="bg-background text-heading-foreground dark:bg-foreground/5">
                <x-tabler-device-mobile-question
                    class="size-6"
                    stroke-width="1.5"
                />
            </x-lqd-icon>
            <h4 class="m-0 flex items-center gap-1 text-base font-medium">
                {{ __('User Client') }}
                <x-info-tooltip text="{{ __('See which devices are being used to access the platform.') }}" />
            </h4>
        </div>
        <x-button
            variant="link"
            href="{{ route('dashboard.admin.users.index') }}"
        >
            <span class="text-nowrap font-bold text-foreground"> {{ __('View User Clients') }} </span>
            <x-tabler-chevron-right class="size-4 rtl:rotate-180" />
        </x-button>
    </x-slot:head>
    @php
        $mobileCount = \App\Helpers\Classes\Helper::appIsDemo() ? 44 : cache()->get('user_behavior_data')[0]['value'];
        $desktopCount = \App\Helpers\Classes\Helper::appIsDemo() ? 63 : cache()->get('user_behavior_data')[1]['value'];

        $all = $mobileCount + $desktopCount;
        if ($all == 0) {
            $all = 1;
        }

        $mobilePercent = round(($mobileCount / $all) * 100, 2);
        $desktopPercent = round(($desktopCount / $all) * 100, 2);
    @endphp
    <div id="user-behaviour-chart">
        <div>
            <div class="lqd-progress flex h-2 overflow-hidden rounded-full">
                @if ($desktopPercent != 0)
                    <div
                        class="lqd-progress-bar h-full grow"
                        style="width: {{ $desktopPercent }}%; background-color: #1FBA96;"
                    >
                    </div>
                @endif
                @if ($mobilePercent != 0)
                    <div
                        class="lqd-progress-bar h-full grow"
                        style="width: {{ $mobilePercent }}%; background-color: #20C69F33;"
                    >
                    </div>
                @endif
            </div>
        </div>

        <div class="flex h-32">
            <div class="group flex shrink-0 grow basis-0 flex-col justify-center space-y-3 px-5 pt-9 text-xs text-heading-foreground sm:px-9 md:max-lg:px-2">
                <div class="flex items-center gap-2 group-last:flex-row-reverse">
                    <span
                        class="h-[18px] w-1 rounded-full"
                        style="background-color: #1FBA96"
                    ></span>
                    {{ __('Desktop') }}
                </div>
                <div class="font-heading text-[28px] font-bold text-heading-foreground/80">
                    {{ number_format($desktopPercent, 2) }}%
                </div>
            </div>
            {{-- begin: criteria --}}
            <div class="relative flex w-px items-center justify-center bg-border">
                <div class="inline-flex size-[50px] shrink-0 items-center justify-center rounded-full border bg-background text-sm font-medium shadow-sm">
                    @lang('vs')
                </div>
            </div>
            {{-- end: criteria --}}
            <div class="group flex shrink-0 grow basis-0 flex-col justify-center space-y-3 px-5 pt-9 text-end text-xs text-heading-foreground sm:px-9 md:max-lg:px-2">
                <div class="flex items-center gap-2 group-last:flex-row-reverse">
                    <span
                        class="h-[18px] w-1 rounded-full"
                        style="background-color: #20C69F33"
                    ></span>
                    {{ __('Mobile') }}
                </div>
                <div class="font-heading text-[28px] font-bold text-heading-foreground/80">
                    {{ number_format($mobilePercent, 2) }}%
                </div>
            </div>
        </div>
    </div>
</x-card>
