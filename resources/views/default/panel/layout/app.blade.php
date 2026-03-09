@php
    use App\Helpers\Classes\MarketplaceHelper;
    $theme = get_theme();
    $is_auth = Auth::check();
    $disable_floating_menu = true;
    $wide_layout_px_class = Theme::getSetting('wideLayoutPaddingX', '');
    $theme_google_fonts = Theme::getSetting('dashboard.googleFonts');
    $sidebarEnabledPages = Theme::getSetting('dashboard.sidebarEnabledPages') ?? [];
    $has_sidebar = in_array(Route::currentRouteName(), $sidebarEnabledPages, true) || (isset($has_sidebar) && $has_sidebar);
    $body_classname = Theme::getSetting('dashboard.bodyClass', '');

    if (isset($body_class)) {
        $body_classname .= ' ' . $body_class;
    }

    if (!empty($wide_layout_px)) {
        $wide_layout_px_class = $wide_layout_px;
    }

    if (Route::currentRouteName()) {
        $route_name = str_replace(['dashboard.', '.index', '.'], ['', '', '-'], Route::currentRouteName());
        $body_classname .= ' page-' . $route_name;
    }
@endphp
<!DOCTYPE html>
<html
    class="scroll-smooth"
    lang="{{ \App\Helpers\Classes\Localization::getLocale() }}"
    dir="{{ \App\Helpers\Classes\Localization::getCurrentLocaleDirection() }}"
>

@include('panel.layout.partials.head')

<body
    data-theme="{{ setting('dash_theme') }}"
    @class([
        @twMerge(
            'group/body bg-background font-body text-xs text-foreground antialiased transition-bg',
            $body_classname),
        'has-sidebar' => $has_sidebar,
        'is-not-auth' => !$is_auth,
        'is-auth' => $is_auth,
        'is-admin-page' =>
            Auth::check() &&
            (Route::is('dashboard.admin*') ||
                Route::is('dashboard.blog*') ||
                Route::is('dashboard.page*')),
        'is-auth-page' => Route::is('login', 'register', 'forgot_password'),
        'hide-navbar' => isset($disable_navbar),
        'hide-footer' => isset($disable_footer),
        'hide-header' => isset($disable_header),
        'hide-titlebar' => isset($disable_titlebar),
    ])
>
    @includeIf('panel.layout.after-body-open-immediate')
    @stack('after-body-open-immediate')

    @if ($app_is_not_demo)
        @includeFirst(['onboarding-pro::banner', 'vendor.empty'])
    @endif

    @include('panel.layout.partials.mode-script')

    @includeIf('panel.layout.after-body-open')
    @stack('after-body-open')

    @include('panel.layout.partials.loading')

    <div class="lqd-page relative flex min-h-full flex-col">

        <div class="lqd-page-wrapper grow-1 flex">
            @php
                $showMenu = auth()->check();
                $isImageProLanding = MarketplaceHelper::isRegistered('ai-image-pro') && request()->is('ai-image-pro*');
                if (!$showMenu && $isImageProLanding) {
                    $showMenu = true;
                }
            @endphp
            @if ($showMenu)
                @if (!isset($disable_navbar))
                    @include('panel.layout.navbar')
                @endif
            @endif
            <div class="lqd-page-content-wrap flex grow flex-col overflow-hidden">
                @if ($good_for_now)
                    @auth
                        @if (!isset($disable_header))
                            @include('panel.layout.header', ['layout_wide', isset($layout_wide) ? $layout_wide : ''])
                        @endif
                        @if (!isset($disable_titlebar))
                            @include('panel.layout.titlebar', ['layout_wide', isset($layout_wide) ? $layout_wide : ''])
                        @endif
                    @endauth

                    @yield('before-content-container')

                    <div @class([
                        'lqd-page-content-container',
                        'h-full',
                        'container' => !isset($layout_wide) || empty($layout_wide),
                        'container-fluid' => isset($layout_wide) && !empty($layout_wide),
                        $wide_layout_px_class =>
                            filled($wide_layout_px_class) &&
                            (isset($layout_wide) && !empty($layout_wide)),
                    ])>

                        @yield('content')

                        @if ($app_is_not_demo)
                            @includeFirst(['onboarding-pro::survey', 'vendor.empty'])
                        @endif
                    </div>
                @elseif(Auth::check() && !$good_for_now && Route::currentRouteName() != 'dashboard.admin.settings.general')
                    <div @class([
                        'lqd-page-content-container',
                        'container' => !isset($layout_wide) || empty($layout_wide),
                        'container-fluid' => isset($layout_wide) && !empty($layout_wide),
                        $wide_layout_px_class =>
                            filled($wide_layout_px_class) &&
                            (isset($layout_wide) && !empty($layout_wide)),
                    ])>
                        @include('vendor.installer.magicai_c4st_Act')
                    </div>
                @else
                    @auth
                        @if (!isset($disable_header))
                            @include('panel.layout.header', ['layout_wide', isset($layout_wide) ? $layout_wide : ''])
                        @endif
                        @if (!isset($disable_titlebar))
                            @include('panel.layout.titlebar', ['layout_wide', isset($layout_wide) ? $layout_wide : ''])
                        @endif
                    @endauth

                    @yield('before-content-container')

                    <div @class([
                        'lqd-page-content-container',
                        'container' => !isset($layout_wide) || empty($layout_wide),
                        'container-fluid' => isset($layout_wide) && !empty($layout_wide),
                        $wide_layout_px_class =>
                            filled($wide_layout_px_class) &&
                            (isset($layout_wide) && !empty($layout_wide)),
                    ])>

                        @yield('content')
                    </div>
                @endif

                @auth
                    @if (!isset($disable_footer))
                        @include('panel.layout.footer')
                    @endif

                    @if ($has_sidebar && (!isset($disable_default_sidebar) || empty($disable_default_sidebar)))
                        @includeIf('panel.layout.sidebar')
                    @endif
                @endauth
            </div>
        </div>
    </div>

    @auth
        @if (!isset($disable_floating_menu))
            <x-floating-menu />
        @endif
        @if (!isset($disable_mobile_bottom_menu))
            <x-bottom-menu />
        @endif
    @endauth

    @if (!isset($disableChatbot))
        @php
            $currentUrl = url()->current();
            $shouldShowChatbot =
                in_array($settings_two->chatbot_status, ['dashboard', 'both']) &&
                !activeRoute('dashboard.user.openai.chat.chat', 'dashboard.user.openai.webchat.workbook', 'dashboard.user.advanced-image.index') &&
                route('dashboard.user.openai.generator.workbook', 'ai_vision') !== $currentUrl &&
                route('dashboard.user.openai.generator.workbook', 'ai_chat_image') !== $currentUrl &&
                route('dashboard.user.openai.generator.workbook', 'ai_pdf') !== $currentUrl;

            if (MarketplaceHelper::isRegistered('ai-chat-pro')) {
                $shouldShowChatbot = $shouldShowChatbot && route('dashboard.user.openai.chat.pro.index') !== $currentUrl && route('chat.pro') !== $currentUrl;
            }
        @endphp

        @includeWhen($shouldShowChatbot, 'panel.chatbot.widget')
    @endif

    @includeIf('live-customizer::particles.customizer')

    @include('panel.layout.scripts')

    @if (session()->has('message'))
        <script>
            toastr.{{ session('type') }}('{{ session('message') }}');
        </script>
    @endif

    @if ($errors->any())
        <script>
            @foreach ($errors->all() as $error)
                toastr.error('{{ $error }}');
            @endforeach
        </script>
    @endif

    @stack('script')

    <script src="{{ custom_theme_url('/assets/js/frontend.js') }}"></script>

    @if ($setting->dashboard_code_before_body != null)
        {!! $setting->dashboard_code_before_body !!}
    @endif

    <script src="{{ custom_theme_url('assets/js/chatbot.js') }}"></script>

    @includeIf('panel.layout.before-body-close')

    @if ($app_is_not_demo)
        @auth()
            @if (auth()->user()->isAdmin())
                <script src="{{ custom_theme_url('/assets/js/panel/update-check.js') }}"></script>
            @endif
        @endauth
        <script src="{{ custom_theme_url('/assets/libs/introjs/intro.min.js') }}"></script>
        @includeIf('seo-tool::particles.generate-seo-script')
        @include('panel.layout.includes.lazy-intercom')
        @include('panel.layout.includes.subscription-status')
    @endif

    @livewireScriptConfig()

    <template id="typing-template">
        <div class="lqd-typing relative inline-flex items-center gap-3 rounded-full bg-secondary !px-3 !py-2 text-xs font-medium leading-none text-secondary-foreground">
            {{ __('Typing') }}
            <div class="lqd-typing-dots flex h-5 items-center gap-1">
                <span class="lqd-typing-dot inline-block !h-1 !w-1 rounded-full !bg-current opacity-40 ![animation-delay:0.2s]"></span>
                <span class="lqd-typing-dot inline-block !h-1 !w-1 rounded-full !bg-current opacity-60 ![animation-delay:0.3s]"></span>
                <span class="lqd-typing-dot inline-block !h-1 !w-1 rounded-full !bg-current opacity-80 ![animation-delay:0.4s]"></span>
            </div>
        </div>
    </template>

    <template id="copy-btns-template">
        <div
            class="pointer-events-none invisible flex translate-y-1 flex-col gap-2 opacity-0 transition-all group-[&.active]/copy-wrap:pointer-events-auto group-[&.active]/copy-wrap:visible group-[&.active]/copy-wrap:translate-y-0 group-[&.active]/copy-wrap:opacity-100">
            <button
                class="group/btn relative inline-flex size-9 items-center justify-center rounded-full bg-white p-0 text-[12px] text-black shadow-lg transition-all hover:scale-110"
                data-copy-type="md"
                type="button"
            >
                <x-tabler-markdown
                    class="size-5"
                    stroke-width="1.5"
                />
                <span
                    class="absolute end-full top-1/2 me-1 inline-block -translate-y-1/2 translate-x-1 whitespace-nowrap rounded-full bg-white px-3 py-1 font-medium leading-5 opacity-0 shadow-lg transition-all group-hover/btn:translate-x-0 group-hover/btn:opacity-100"
                >
                    @lang('Copy Markdown')
                </span>
            </button>
            <button
                class="group/btn relative inline-flex size-9 items-center justify-center rounded-full bg-white p-0 text-[12px] text-black shadow-lg transition-all hover:scale-110"
                data-copy-type="html"
                type="button"
            >
                <x-tabler-file-type-html
                    class="size-5"
                    stroke-width="1.5"
                />
                <span
                    class="absolute end-full top-1/2 me-1 inline-block -translate-y-1/2 translate-x-1 whitespace-nowrap rounded-full bg-white px-3 py-1 font-medium leading-5 opacity-0 shadow-lg transition-all group-hover/btn:translate-x-0 group-hover/btn:opacity-100"
                >
                    @lang('Copy HTML')
                </span>
            </button>
        </div>
    </template>

    @if ($app_is_demo)
        <x-demo-switcher themes-type="Dashboard" />
    @endif

    @includeIf('content-manager::media-modal')
</body>

</html>
