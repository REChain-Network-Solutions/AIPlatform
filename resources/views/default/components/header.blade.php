<header class="lqd-header relative flex h-[--header-height] border-b border-header-border bg-header-background text-xs font-medium transition-colors max-lg:z-40">
    <div @class([
        'lqd-header-container flex w-full grow gap-2 px-4 max-lg:w-full max-lg:max-w-none',
        'container' => !$attributes->get('layout-wide'),
        'container-fluid' => $attributes->get('layout-wide'),
        Theme::getSetting('wideLayoutPaddingX', '') =>
            filled(Theme::getSetting('wideLayoutPaddingX', '')) &&
            $attributes->get('layout-wide'),
    ])>

        {{-- Mobile nav toggle and logo --}}
        <div class="mobile-nav-logo flex items-center gap-3 lg:hidden">
            <button
                class="lqd-mobile-nav-toggle flex size-10 items-center justify-center"
                type="button"
                x-init
                @click.prevent="$store.mobileNav.toggleNav()"
                :class="{ 'lqd-is-active': !$store.mobileNav.navCollapse }"
            >
                <span class="lqd-mobile-nav-toggle-icon relative h-[2px] w-5 rounded-xl bg-current"></span>
            </button>
            <x-header-logo />
        </div>

        {{-- Title slot --}}
        @if ($title ?? false)
            <div class="header-title-container peer/title hidden items-center lg:flex">
                <h1 class="m-0 font-semibold">
                    {{ $title }}
                </h1>
            </div>
        @endif

        @includeFirst(['focus-mode::header', 'components.includes.ai-tools', 'vendor.empty'])

        {{-- Search form --}}
        <div class="header-search-container hidden items-center peer-[&.header-title-container]/title:grow peer-[&.header-title-container]/title:justify-center lg:flex">
            <x-header-search class:input="ps-10" />
        </div>

        <div class="header-actions-container flex grow justify-end gap-4 max-lg:basis-2/3 max-lg:gap-2">

            {{-- Action buttons --}}
            @if ($actions ?? false)
                {{ $actions }}
            @else
                <div class="flex items-center max-xl:gap-2 max-lg:hidden xl:gap-3">
                    @if (Auth::user()?->isAdmin())
                        @if ($app_is_not_demo)
                            <x-update-available />
                        @endif
                        <x-button
                            href="{{ route('dashboard.admin.index') }}"
                            variant="ghost-shadow"
                        >
                            {{ __('Admin Panel') }}
                        </x-button>
                    @endif

                    @if ($settings_two->liquid_license_type == 'Extended License' && $app_is_demo)
                        @if ($subscription = getSubscription())
                            <x-button
                                class="max-xl:hidden"
                                href="{{ route('dashboard.user.payment.subscription') }}"
                                variant="ghost-shadow"
                            >
                                {{ $subscription?->plan?->name }} - {{ getSubscriptionDaysLeft() }}
                                {{ __('Days Left') }}
                            </x-button>
                        @else
                            <x-button
                                class="max-xl:hidden"
                                href="{{ route('dashboard.user.payment.subscription') }}"
                                variant="ghost-shadow"
                            >
                                {{ __('No Active Subscription') }}
                            </x-button>
                        @endif

                        <x-button
                            class="max-xl:hidden"
                            href="{{ route('dashboard.user.payment.subscription') }}"
                        >
                            <x-tabler-bolt class="size-4 fill-current" />
                            <span class="max-lg:hidden">
                                {{ __('Upgrade') }}
                            </span>
                        </x-button>
                    @endif
                </div>
            @endif

            <div class="flex items-center gap-4 max-lg:gap-2">
                @includeIf('marketing-bot::header.inbox-notification')

                @includeIf('chatbot-agent::header.inbox-notification')

                @includeIf('social-media-agent::notifications.notifications-drawer')

                {{-- Dark/light switch --}}
                @if (Theme::getSetting('dashboard.supportedColorSchemes') === 'all')
                    <x-light-dark-switch />
                @endif

                @includeFirst(['focus-mode::ai-tools-button', 'components.includes.ai-tools-button', 'vendor.empty'])

                @if ($app_is_not_demo && Auth::user()->isAdmin() && !\App\Helpers\Classes\Helper::showIntercomForVipMembership())
                    {{-- Upgrade button --}}
                    <x-modal
                        class="max-lg:hidden"
                        type="page"
                    >
                        <x-slot:trigger
                            custom
                        >
                            <x-button
                                class="lqd-header-upgrade-btn flex items-center justify-center p-0 text-current"
                                href="#"
                                variant="link"
                                title="{{ __('Premium Membership') }}"
                                @click.prevent="toggleModal()"
                            >
                                <x-tabler-bolt
                                    class="size-[22px]"
                                    stroke-width="1.5"
                                />
                            </x-button>
                        </x-slot:trigger>
                        <x-slot:modal>
                            @includeIf('premium-support.index')
                        </x-slot:modal>
                    </x-modal>
                @endif

                @if (setting('notification_active', 0))
                    {{-- Notifications --}}
                    <x-notifications />
                @endif

                {{-- Language dropdown --}}
                @if (count(explode(',', $settings_two->languages)) > 1)
                    <x-language-dropdown />
                @endif

                {{-- User menu --}}
                <x-user-dropdown />
            </div>
        </div>
    </div>
</header>
