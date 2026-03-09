<a
    class="relative z-1 block px-0"
    href="{{ route('dashboard.index') }}"
>
    @if (isset($setting->logo_dashboard))
        <img
            class="h-auto w-full group-[.navbar-shrinked]/body:hidden dark:hidden"
            src="{{ custom_theme_url($setting->logo_dashboard_path, true) }}"
            @if (isset($setting->logo_dashboard_2x_path) && !empty($setting->logo_dashboard_2x_path)) srcset="/{{ $setting->logo_dashboard_2x_path }} 2x" @endif
            alt="{{ $setting->site_name }}"
        >
        <img
            class="hidden h-auto w-full group-[.navbar-shrinked]/body:hidden dark:block"
            src="{{ custom_theme_url($setting->logo_dashboard_dark_path, true) }}"
            @if (isset($setting->logo_dashboard_dark_2x_path) && !empty($setting->logo_dashboard_dark_2x_path)) srcset="/{{ $setting->logo_dashboard_dark_2x_path }} 2x" @endif
            alt="{{ $setting->site_name }}"
        >
    @else
        <img
            class="h-auto w-full group-[.navbar-shrinked]/body:hidden dark:hidden"
            src="{{ custom_theme_url($setting->logo_path, true) }}"
            @if (isset($setting->logo_2x_path) && !empty($setting->logo_2x_path)) srcset="/{{ $setting->logo_2x_path }} 2x" @endif
            alt="{{ $setting->site_name }}"
        >
        <img
            class="hidden h-auto w-full group-[.navbar-shrinked]/body:hidden dark:block"
            src="{{ custom_theme_url($setting->logo_dark_path, true) }}"
            @if (isset($setting->logo_dark_2x_path) && !empty($setting->logo_dark_2x_path)) srcset="/{{ $setting->logo_dark_2x_path }} 2x" @endif
            alt="{{ $setting->site_name }}"
        >
    @endif

    <!-- collapsed -->
    <img
        class="mx-auto hidden h-auto w-full max-w-10 group-[.navbar-shrinked]/body:block dark:!hidden"
        src="{{ custom_theme_url($setting->logo_collapsed_path, true) }}"
        @if (isset($setting->logo_collapsed_2x_path) && !empty($setting->logo_collapsed_2x_path)) srcset="/{{ $setting->logo_collapsed_2x_path }} 2x" @endif
        alt="{{ $setting->site_name }}"
    >
    <img
        class="mx-auto hidden h-auto w-full max-w-10 group-[.theme-dark.navbar-shrinked]/body:block"
        src="{{ custom_theme_url($setting->logo_collapsed_dark_path, true) }}"
        @if (isset($setting->logo_collapsed_dark_2x_path) && !empty($setting->logo_collapsed_dark_2x_path)) srcset="/{{ $setting->logo_collapsed_dark_2x_path }} 2x" @endif
        alt="{{ $setting->site_name }}"
    >

</a>
