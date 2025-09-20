@php
    $in_cart = in_array($item['id'], $cartExists);
@endphp

@if (!$item['licensed'] && $item['price'] && $item['is_buy'] && !$item['only_show'])
    @if ($app_is_not_demo)
        @if (isset($item['parent']))
            @if (\App\Helpers\Classes\MarketplaceHelper::isRegistered($item['parent']['slug']))
                @if ($item['only_premium'])
                    <a
                        @class([
                            'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:bg-text-white',
                            'in-cart' => $in_cart,
                        ])
                        @if ($item['check_subscription']) data-toogle="cart"
							href="{{ route('dashboard.admin.marketplace.cart.add-delete', $item['id']) }}"
						@else
							onclick="return toastr.info('This extension is for premium customers only.')" @endif
                    >
                        <x-tabler-shopping-cart
                            class="size-6"
                            stroke-width="1.5"
                        />
                    </a>
                @else
                    <a
                        data-toogle="cart"
                        @class([
                            'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:bg-text-white',
                            'in-cart' => $in_cart,
                        ])
                        href="{{ route('dashboard.admin.marketplace.cart.add-delete', $item['id']) }}"
                    >
                        <x-tabler-shopping-cart
                            class="size-6"
                            stroke-width="1.5"
                        />
                    </a>
                @endif
            @else
                <div
                    @class([
                        'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:bg-text-white',
                        'in-cart' => $in_cart,
                    ])
                    onclick="return toastr.info('{{ $item['parent']['message'] }}')"
                >
                    <a href="#">
                        <x-tabler-shopping-cart
                            class="size-6"
                            stroke-width="1.5"
                        />
                    </a>
                </div>
            @endif
        @else
            @if ($item['only_premium'])
                <a
                    @class([
                        'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:bg-text-white',
                        'in-cart' => $in_cart,
                    ])
                    @if ($item['check_subscription']) data-toogle="cart"
						href="{{ route('dashboard.admin.marketplace.cart.add-delete', $item['id']) }}"
					@else
						href="#"
						onclick="return toastr.info('This extension is for premium customers only.')" @endif
                >
                    <x-tabler-shopping-cart
                        class="size-6"
                        stroke-width="1.5"
                    />
                </a>
            @else
                <a
                    data-toogle="cart"
                    @class([
                        'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:bg-text-white',
                        'in-cart' => $in_cart,
                    ])
                    href="{{ route('dashboard.admin.marketplace.cart.add-delete', $item['id']) }}"
                >
                    <x-tabler-shopping-cart
                        class="size-6"
                        stroke-width="1.5"
                    />
                </a>
            @endif
        @endif
    @endif
@endif
