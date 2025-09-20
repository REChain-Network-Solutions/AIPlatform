@php
    $in_cart = in_array($item['id'], $cartExists);
@endphp

@if (!$item['licensed'] && $item['price'] && $item['is_buy'])
    @if (isset($item['parent']))

        @if (\App\Helpers\Classes\MarketplaceHelper::isRegistered($item['parent']['slug']))

            @if (in_array($item['slug'], ['whatsapp', 'telegram', 'facebook', 'instagram']))
                @if (\App\Helpers\Classes\MarketplaceHelper::getDbVersion($item['parent']['slug']) >= $item['parent']['min_version'])
                    @if ($item['only_premium'])
                        @if ($item['check_subscription'])
                            <x-button
                                data-toogle="cart"
                                @class([
                                    'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:text-white',
                                    'in-cart' => $in_cart,
                                ])
                                href="{{ route('dashboard.admin.marketplace.cart.add-delete', $item['id']) }}"
                                variant="outline"
                            >
                                <x-tabler-shopping-cart
                                    class="size-6"
                                    stroke-width="1.5"
                                />
                            </x-button>
                        @endif
                    @else
                        <x-button
                            data-toogle="cart"
                            @class([
                                'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:text-white',
                                'in-cart' => $in_cart,
                            ])
                            href="{{ route('dashboard.admin.marketplace.cart.add-delete', $item['id']) }}"
                            variant="outline"
                        >
                            <x-tabler-shopping-cart
                                class="size-6"
                                stroke-width="1.5"
                            />
                        </x-button>
                    @endif
                @else
                    <x-button
                        @class([
                            'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:text-white',
                            'in-cart' => $in_cart,
                        ])
                        variant="outline"
                        href="#"
                        onclick="return toastr.info('{{ $item['parent']['min_version_message'] }}')"
                    >
                        <x-tabler-shopping-cart
                            class="size-6"
                            stroke-width="1.5"
                        />
                    </x-button>
                @endif
            @else
                @if ($item['only_premium'])
                    @if ($item['check_subscription'])
                        <x-button
                            data-toogle="cart"
                            @class([
                                'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:text-white',
                                'in-cart' => $in_cart,
                            ])
                            href="{{ route('dashboard.admin.marketplace.cart.add-delete', $item['id']) }}"
                            variant="outline"
                        >
                            <x-tabler-shopping-cart
                                class="size-6"
                                stroke-width="1.5"
                            />
                        </x-button>
                    @endif
                @else
                    <x-button
                        data-toogle="cart"
                        @class([
                            'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:text-white',
                            'in-cart' => $in_cart,
                        ])
                        href="{{ route('dashboard.admin.marketplace.cart.add-delete', $item['id']) }}"
                        variant="outline"
                    >
                        <x-tabler-shopping-cart
                            class="size-6"
                            stroke-width="1.5"
                        />
                    </x-button>
                @endif
            @endif
        @else
            <x-button
                @class([
                    'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:text-white',
                    'in-cart' => $in_cart,
                ])
                onclick="return toastr.info('{{ $item['parent']['message'] }}')"
                variant="outline"
                href="#"
            >
                <x-tabler-shopping-cart
                    class="size-6"
                    stroke-width="1.5"
                />
            </x-button>

        @endif
    @else
        @if ($item['only_premium'])
            @if ($item['check_subscription'])
                <x-button
                    data-toogle="cart"
                    @class([
                        'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:text-white',
                        'in-cart' => $in_cart,
                    ])
                    href="{{ route('dashboard.admin.marketplace.cart.add-delete', $item['id']) }}"
                    variant="outline"
                >
                    <x-tabler-shopping-cart
                        class="size-6"
                        stroke-width="1.5"
                    />
                </x-button>
            @endif
        @else
            <x-button
                data-toogle="cart"
                @class([
                    'relative z-1 inline-grid size-11 shrink-0 place-items-center rounded border transition hover:bg-foreground/10 [&.in-cart]:bg-emerald-500 [&.in-cart]:text-white [&.updating-cart]:pointer-events-none [&.updating-cart]:opacity-50 [&.in-cart]:hover:bg-red-500 [&.in-cart]:hover:text-white',
                    'in-cart' => $in_cart,
                ])
                href="{{ route('dashboard.admin.marketplace.cart.add-delete', $item['id']) }}"
                variant="outline"
            >
                <x-tabler-shopping-cart
                    class="size-6"
                    stroke-width="1.5"
                />
            </x-button>
        @endif
    @endif

@endif
