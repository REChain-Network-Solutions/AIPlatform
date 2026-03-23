@php
    $base_class = 'lqd-tooltip-container group relative inline-flex cursor-default';
    $content_class =
        'lqd-tooltip-content invisible fixed z-50 w-max max-w-72 text-pretty rounded-xl bg-background/80 px-4 py-3 text-center text-xs leading-normal text-foreground opacity-0 shadow-lg shadow-black/5 backdrop-blur-sm backdrop-saturate-150 transition';
@endphp

<span
    {{ $attributes->withoutTwMergeClasses()->twMerge($base_class, $attributes->get('class')) }}
    x-data="lqdInfoTooltip({ preferredPlacement: '{{ $anchor === 'top' ? 'bottom' : 'top' }}' })"
    x-ref="trigger"
    @mouseenter="show()"
    @mouseleave="hide()"
    @focusin="show()"
    @focusout="hide()"
>
    <span {{ $attributes->twMergeFor('icon', 'lqd-tooltip-icon opacity-40') }}>
        <x-tabler-info-circle-filled class="size-4" />
    </span>
    <span
        {{ $attributes->twMergeFor('content', $content_class) }}
        x-ref="tooltip"
        :class="{ 'invisible opacity-0': !open }"
    >
        {{ $text }}

        @if ($drivers->isNotEmpty())
            <div>
                <h5 class="font-semibold">{{ __('Credits Details') }}</h5>
                <hr class="my-3 border-heading-foreground/10" />

                @foreach ($drivers as $driver)
                    @if (!$driver->hasCreditBalance())
                        @continue
                    @endif
                    <div class="flex justify-between gap-x-1 border-b py-1.5 text-2xs last:border-b-0">
                        <span class="text-start">{{ $driver->enum()->value }}</span>
                        <span class="text-end font-medium">{{ $driver->isUnlimitedCredit() ? __('Unlimited') : $driver->creditBalance() }}</span>
                    </div>
                @endforeach
            </div>
        @endif
    </span>
</span>
