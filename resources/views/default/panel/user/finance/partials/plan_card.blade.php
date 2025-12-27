@php($newDiscountedPrice = $newDiscountedPrice ?? 0)

<div @class([
    'w-full rounded-3xl border bg-background',
    'shadow-[0_7px_20px_rgba(0,0,0,0.04)]' => $plan->is_featured,
])>
    <div class="flex h-full flex-col p-7">

        <h2 class="mb-5 flex items-start leading-none text-heading-foreground">
            @lang('Order Summary')
        </h2>
        <div class="inline-flex flex-col items-start gap-2">
            <p class="mb-0 mt-1 flex items-start leading-none text-heading-foreground">
                {{ __($plan->name) }} / {{ $plan->type == 'prepaid' ? __('One time') : __(formatCamelCase($plan->frequency)) }} @lang('Plan')
            </p>
            @if ($plan->is_featured == 1)
                <div class="inline-flex rounded-full bg-gradient-to-r from-[#ece7f7] via-[#e7c5e6] to-[#e7ebf9] px-3 py-1 text-3xs text-black">
                    {{ __('Popular plan') }}
                </div>
            @endif
        </div>

        <ul class="list-unstyled mt-5 w-full rounded-lg border p-4">
            <li class="flex px-1">
                <p class="grow text-start">{{ __('Subtotal') }}</p>
                <p class="whitespace-nowrap text-end">{!! displayCurr(currency()->symbol, $plan->price, 0, null) !!}</p>
            </li>
            @if ($plan->price - $newDiscountedPrice > 0)
                <hr class="mt-0">
                <li class="flex px-1">
                    <div class="flex grow text-start">
                        <x-tabler-tag class="me-1 size-5 -scale-x-100 pt-[0.15rem]" />
                        <p>{{ __('Discount') }}</p> {!! displayDiscountOrCouponName($plan) !!}
                    </div>
                    <p class="whitespace-nowrap text-end text-red-600">- {!! displayCurr(currency()->symbol, $plan->price - $newDiscountedPrice, null) !!}</p>
                </li>
            @endif
            <hr class="mt-0">
            <li class="flex px-1">
                <p class="grow text-start">{{ __('Tax') }} ({{ $taxRate ?? 0 }}% VAT)</p>
                <p class="whitespace-nowrap text-end">{!! displayCurr(currency()->symbol, $taxValue ?? 0, tax_included: $plan->price_tax_included) !!}</p>
            </li>
            <hr class="mt-0">
            <li class="flex px-1">
                <p class="grow text-start">{{ __('Total') }}</p>
                <p class="whitespace-nowrap text-end">{!! displayCurr(currency()->symbol, $plan->price, $taxValue ?? 0, $newDiscountedPrice ?? null, tax_included: $plan->price_tax_included) !!}</p>
            </li>
        </ul>

        <x-plan-details-card
            :plan="$plan"
            :period="$plan->frequency"
        />
        <div class="mt-7 text-center">
            <a
                class="btn w-full rounded-md p-[1.15em_2.1em] text-[15px] group-[.theme-dark]/body:!bg-[rgba(255,255,255,1)] group-[.theme-dark]/body:!text-[rgba(0,0,0,0.9)]"
                href="{{ auth()->check() ? route('dashboard.user.payment.subscription') : route('index') . '#pricing' }}"
            >{{ __('Change Plan') }}</a>
        </div>
    </div>
</div>
