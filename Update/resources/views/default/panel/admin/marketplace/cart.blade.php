@extends('panel.layout.app')
@section('title', 'Cart')
@section('titlebar_actions', '')
@section('titlebar_pretitle', '')
@section('content')
    <div class="py-10">
        <div class="mx-auto w-full text-center lg:w-6/12 lg:px-9">

            @if (isset($items['data']) && $items['data'] && $app_is_not_demo)
                <h2 class="mb-4">
                    {{ __('Cart Items') }}
                </h2>

                <p class="mx-auto mb-8 lg:w-10/12">
                    {{ __('To complete the order, make payment') }}.
                </p>
                <div class="mx-auto mb-4 rounded-lg border text-heading-foreground">
                    @foreach ($items['data'] as $item)
                        @php
                            $in_cart = true;
                        @endphp

                        <div
                            class="flex items-center justify-between gap-2 border-b px-4 py-3"
                            id="ext-{{ $item['extension_id'] }}"
                        >
                            <p class="mb-0 text-start">
                                {{ $item['extension']['name'] }}
                            </p>
                            <x-button
                                data-toogle="cart"
                                data-delete-item="ext-{{ $item['extension_id'] }}"
                                variant="outline"
                                @class([
                                    'relative z-1 shrink-0 hover:bg-foreground/10 [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:text-white',
                                    'in-cart' => $in_cart,
                                ])
                                href="{{ route('dashboard.admin.marketplace.cart.add-delete', $item['extension_id']) }}"
                                size="sm"
                            >
                                {{ $item['extension']['price'] . ' USD' }}
                                <span
                                    class="inline-grid size-7 place-items-center rounded-full border group-hover:border-background group-hover:bg-background group-hover:text-foreground"
                                >
                                    <x-tabler-trash class="size-4" />
                                </span>
                            </x-button>
                        </div>
                    @endforeach
                    <form
                        class="border-b p-4"
                        method="POST"
                        action="{{ route('dashboard.admin.marketplace.cart.coupon') }}"
                    >
                        @csrf
                        <div class="flex items-center gap-2">
                            <x-forms.input
                                class:container="grow"
                                id="coupon_code"
                                value="{{ isset($items['coupon']['code']) ? $items['coupon']['code'] : '' }}"
                                type="text"
                                name="coupon_code"
                                size="lg"
                                placeholder="{{ __('Enter coupon code') }}"
                            />
                            @if (isset($items['coupon']['code']))
                                <x-button
                                    class="shrink-0"
                                    id="apply_coupon_btn"
                                    href="{{ route('dashboard.admin.marketplace.cart.delete.coupon') }}"
                                    variant="danger"
                                >
                                    {{ __('Remove Coupon') }}
                                </x-button>
                            @else
                                <x-button
                                    class="shrink-0"
                                    class="whitespace-nowrap"
                                    id="apply_coupon_btn"
                                    type="submit"
                                >
                                    {{ __('Apply Coupon') }}
                                </x-button>
                            @endif
                        </div>
                        @error('coupon_code')
                            <div
                                class="mt-2 text-sm text-red-500"
                                id="coupon_message"
                            >
                                {{ $message }}
                            </div>
                        @enderror
                    </form>

                    <!-- Total Section -->
                    <div class="bg-foreground/5 p-4">
                        <div
                            class="mb-2 flex items-center justify-between"
                            id="subtotal_row"
                        >
                            <span class="font-medium">{{ __('Subtotal') }}:</span>
                            <span id="subtotal_amount">${{ number_format($items['sub_total_price'], 2) }} USD</span>
                        </div>
                        @if ($items['coupon'])
                            <div class="mb-2 flex items-center justify-between text-green-600">
                                <span>{{ __('Discount') }} (<span id="coupon_name">{{ data_get($items, 'coupon.code') }}</span>):</span>
                                <span id="discount_amount">-${{ data_get($items, 'coupon.discount_value') }} USD</span>
                            </div>
                        @endif

                        <hr class="my-2">
                        <div class="flex items-center justify-between text-lg font-bold">
                            <span>{{ __('Total') }}:</span>
                            <span id="total_amount">${{ number_format($items['total_price'], 2) }} USD</span>
                        </div>
                    </div>
                </div>

                <p class="mb-7 opacity-60">
                    {{ __('Tax included. Your payment is secured via SSL.') }}
                </p>

                <div id="checkout">
                    <x-button
                        class="w-full"
                        target="_blank"
                        size="lg"
                        href="{{ $items['paymentJson'] }}"
                    >
                        {{ __('Pay Now') }}
                    </x-button>
                </div>
            @else
                @if ($app_is_demo)
                    <p class="mb-7 opacity-60">
                        {{ __('You can not add any extensions in demo mode.') }}
                    </p>
                @else
                    <p class="mb-7 opacity-60">
                        {{ __('You did not add any extensions.') }}
                    </p>
                @endif
            @endif
        </div>
    </div>
@endsection

@push('script')
    <script>
        // Cart item removal functionality
        $('[data-toogle="cart"]').on('click', function(event) {
            event.preventDefault();

            const $button = $(this);
            const url = $button.attr('href');

            toastr.info('{{ __('Updating Cart') }}');

            $.get(url, function(data) {
                $button.toggleClass('in-cart');

                $('#itemCount').html(data.itemCount);

                toastr.success(data.message);

                setTimeout(function() {
                    location.reload();
                }, 500);
            });
        });
    </script>
@endpush
