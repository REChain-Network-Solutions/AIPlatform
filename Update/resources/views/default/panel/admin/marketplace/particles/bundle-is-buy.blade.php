@if ($item['is_buy'])
    @if (isset($item['parent']))
        @if (\App\Helpers\Classes\MarketplaceHelper::isRegistered($item['parent']['slug']))
            @if (in_array($item['slug'], ['whatsapp', 'telegram', 'facebook', 'instagram']))
                @if (\App\Helpers\Classes\MarketplaceHelper::getDbVersion($item['parent']['slug']) >= $item['parent']['min_version'])
                    @if ($item['only_premium'])
                        @if ($item['check_subscription'])
                            <x-button
                                class="w-full shadow-xl shadow-black/10"
                                variant="success"
                                target="_blank"
                                size="lg"
                                href="{{ $item['routes']['payment'] }}"
                            >
                                {{ __('Purchase Bundle') }}
                            </x-button>
                        @else
                            <x-button
                                class="w-full shadow-xl shadow-black/10"
                                variant="success"
                                size="lg"
                                href="{{ $marketSubscription['extensionPayment'] ?? '' }}"
                                onclick="return toastr.info('This extension is for premium customers only.')"
                            >
                                {{ __('Only Premium Customer') }}
                            </x-button>
                        @endif
                    @else
                        <x-button
                            class="w-full shadow-xl shadow-black/10"
                            variant="success"
                            target="_blank"
                            size="lg"
                            href="{{ $item['routes']['payment'] }}"
                        >
                            {{ __('Purchase Bundle') }}
                        </x-button>
                    @endif
                @else
                    <x-button
                        class="w-full shadow-xl shadow-black/10"
                        variant="success"
                        size="lg"
                        href="#"
                        onclick="return toastr.info('{{ $item['parent']['min_version_message'] }}')"
                    >
                        {{ __('Purchase Bundle') }}
                    </x-button>
                @endif
            @else
                @if ($item['only_premium'])
                    @if ($item['check_subscription'])
                        <x-button
                            class="w-full shadow-xl shadow-black/10"
                            variant="success"
                            target="_blank"
                            size="lg"
                            href="{{ $item['routes']['payment'] }}"
                        >
                            {{ __('Purchase Bundle') }}
                        </x-button>
                    @else
                        <x-button
                            class="w-full shadow-xl shadow-black/10"
                            variant="success"
                            size="lg"
                            href="{{ $marketSubscription['extensionPayment'] ?? '' }}"
                            onclick="return toastr.info('This extension is for premium customers only.')"
                        >
                            {{ __('Only Premium Customer') }}
                        </x-button>
                    @endif
                @else
                    <x-button
                        class="w-full shadow-xl shadow-black/10"
                        variant="success"
                        target="_blank"
                        size="lg"
                        href="{{ $item['routes']['payment'] }}"
                    >
                        {{ __('Purchase Bundle') }}
                    </x-button>
                @endif
            @endif
        @else
            <x-button
                class="w-full shadow-xl shadow-black/10"
                variant="success"
                size="lg"
                href="#"
                onclick="return toastr.info('{{ $item['parent']['message'] }}')"
            >
                {{ __('Purchase Bundle') }}
            </x-button>
        @endif
    @else
        @if ($item['only_premium'])
            @if ($item['check_subscription'])
                <x-button
                    class="w-full shadow-xl shadow-black/10"
                    variant="success"
                    target="_blank"
                    size="lg"
                    href="{{ $item['routes']['payment'] }}"
                >
                    {{ __('Purchase Bundle') }}
                </x-button>
            @else
                <x-button
                    class="w-full shadow-xl shadow-black/10"
                    size="lg"
                    href="{{ $marketSubscription['extensionPayment'] ?? '' }}"
                    onclick="return toastr.info('This extension is for premium customers only.')"
                >
                    {{ __('Only Premium Customer') }}
                </x-button>
            @endif
        @else
            <x-button
                class="w-full shadow-xl shadow-black/10"
                variant="success"
                target="_blank"
                size="lg"
                href="{{ $item['routes']['payment'] }}"
            >
                {{ __('Purchase Bundle') }}
            </x-button>
        @endif
    @endif
@else
    <span class="lqd-tooltip-container group relative inline-flex cursor-default before:absolute before:-start-1.5 before:-top-1.5 before:h-7 before:w-7">
        <span class="lqd-tooltip-icon">
            <x-button
                class="lqd-tooltip-icon w-full shadow-xl shadow-black/10"
                target="_blank"
                size="lg"
                type="button"
            >
                {{ __('Learn more') }}
            </x-button>
        </span>
        <span
            class="lqd-tooltip-content invisible absolute bottom-full start-1/2 z-50 mb-3 min-w-64 -translate-x-1/2 translate-y-1 rounded-xl bg-background/80 px-4 py-3 text-xs leading-normal text-foreground opacity-0 shadow-lg shadow-black/5 backdrop-blur-sm backdrop-saturate-150 transition-all before:absolute before:inset-x-0 before:-bottom-3 before:h-3 group-hover:visible group-hover:translate-y-0 group-hover:opacity-100"
        >
            {{ __('Contact us for more information:') }}
            <span class="mt-1 flex justify-center">
                info@liquid-themes.com

                <x-tabler-copy
                    class="ms-2"
                    id="copyButton"
                />
            </span>
        </span>
    </span>
@endif
