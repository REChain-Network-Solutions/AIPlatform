<footer class="site-footer lg:pt-30 relative pt-20">
    <figure
        class="pointer-events-none absolute h-full w-full"
        aria-hidden="true"
    >
        <img
            class="absolute start-1/2 top-0 z-0 aspect-square h-full w-full -translate-x-1/2 -translate-y-1/4"
            width="3110"
            height="1142"
            src="{{ custom_theme_url('/assets/landing-page/glow-1.png') }}"
            alt="{{ __('Glowing blob') }}"
        />
    </figure>

    <div class="relative">
        <div class="container mb-24">
            <div class="mx-auto w-2/3 text-center max-lg:w-10/12 max-sm:w-full">
                <h6
                    class="relative mb-16 inline-flex translate-y-6 items-center gap-1.5 overflow-hidden rounded-full bg-secondary px-5 py-2 text-secondary-foreground shadow-xs shadow-primary">
                    <x-tabler-rocket
                        class="size-5"
                        stroke-width="1.5"
                    />
                    {{ __($fSetting->footer_text_small) }}
                </h6>
                <p
                    class="-from-[5%] mb-10 inline-block bg-gradient-to-br from-heading-foreground from-30% to-heading-foreground/70 bg-clip-text font-heading text-[82px] font-bold leading-none tracking-tight text-transparent max-sm:text-[18vw]">
                    {{ __($fSetting->footer_header) }}
                </p>
                <p class="mx-auto mb-12 w-full font-heading text-xl/7 text-heading-foreground/70 lg:w-8/12">
                    {{ __($fSetting->footer_text) }}
                </p>
                <div class="flex flex-wrap items-center justify-center gap-8 text-sm">
                    <a
                        class="relative inline-flex w-56 gap-3 overflow-hidden whitespace-nowrap rounded-lg bg-gradient-to-r from-gradient-from to-gradient-to to-50% py-5 font-semibold text-primary-foreground transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-primary/20"
                        href="{{ !empty($fSetting->footer_button_url) ? $fSetting->footer_button_url : '#' }}"
                        target="_blank"
                    >
                        <span
                            class="flex animate-marquee justify-between gap-3 before:content-[attr(data-txt)] after:content-[attr(data-txt)]"
                            data-txt="{!! __($fSetting->footer_button_text) !!}"
                        >
                            {!! __($fSetting->footer_button_text) !!}
                        </span>
                        <span
                            class="absolute start-3 top-5 flex animate-marquee-2 justify-between gap-3 before:content-[attr(data-txt)] after:content-[attr(data-txt)]"
                            data-txt="{!! __($fSetting->footer_button_text) !!}"
                        >
                            {!! __($fSetting->footer_button_text) !!}
                        </span>
                    </a>
                    <a
                        class="group/btn flex items-center gap-2 text-white transition-colors hover:text-primary"
                        href="{{ !empty($fSetting->footer_button_url) ? $fSetting->footer_button_url : '#' }}"
                        target="_blank"
                    >
                        @lang('Join ' . $setting->site_name)
                        <x-tabler-chevron-right class="size-5 transition-transform group-hover/btn:translate-x-1" />
                    </a>
                </div>
            </div>
        </div>

        <div class="container mb-20">
            <div class="grid grid-cols-2 gap-4 md:grid-cols-2 lg:grid-cols-4">
                @foreach (\App\Models\SocialMediaAccounts::where('is_active', true)->get() as $social)
                    <a
                        class="group flex flex-col items-center rounded-2xl px-4 pb-8 pt-10 text-center leading-tight text-heading-foreground/50 transition-all hover:scale-105 hover:text-heading-foreground"
                        href="#"
                    >
                        <span class="mb-8 block">
                            {!! $social['icon'] !!}
                        </span>
                        <span class="mb-2.5 block font-heading text-lg">
                            {{ $social['title'] }}
                        </span>
                        <span class="block text-2xs transition-opacity group-hover:opacity-60">
                            {{ $social['subtitle'] }}
                        </span>
                    </a>
                @endforeach
            </div>
        </div>

        <hr class="border-white border-opacity-15">

        <div class="container">
            <div class="flex flex-wrap items-center justify-between gap-8 py-14 max-sm:justify-center">
                <a href="{{ route('index') }}">
                    @if (isset($setting->logo_2x_path))
                        <img
                            src="{{ custom_theme_url($setting->logo_path, true) }}"
                            srcset="/{{ $setting->logo_2x_path }} 2x"
                            alt="{{ $setting->site_name }} logo"
                        >
                    @else
                        <img
                            src="{{ custom_theme_url($setting->logo_path, true) }}"
                            alt="{{ $setting->site_name }} logo"
                        >
                    @endif
                </a>

                <ul class="flex flex-wrap items-center justify-end gap-7 text-sm max-sm:justify-center">
                    @foreach (\App\Models\Page::where(['status' => 1, 'show_on_footer' => 1])->get() ?? [] as $page)
                        <li>
                            <a
                                class="inline-flex items-center gap-2"
                                href="/page/{{ $page->slug }}"
                            >
                                {{ $page->title }}
                            </a>
                        </li>
                    @endforeach
                </ul>
            </div>

            <hr class="border-white border-opacity-15">

            <div class="flex flex-wrap items-center justify-between gap-y-4 py-9">
                <p
                    class="w-full text-sm opacity-60 lg:w-1/2"
                    style="color: {{ $fSetting->footer_text_color }};"
                >
                    {{ date('Y') . ' ' . $setting->site_name . '. ' . __($fSetting->footer_copyright) }}
                </p>

                <ul class="flex w-full flex-wrap items-center gap-7 text-sm lg:w-1/2 lg:justify-end">
                    @foreach (\App\Models\SocialMediaAccounts::where('is_active', true)->get() as $social)
                        <li>
                            <a
                                class="inline-flex items-center gap-2"
                                href="{{ $social['link'] }}"
                            >
                                <span class="w-3.5 [&_svg]:h-auto [&_svg]:w-full">
                                    {!! $social['icon'] !!}
                                </span>
                                {{ $social['title'] }}
                            </a>
                        </li>
                    @endforeach
                </ul>
            </div>

        </div>
    </div>
</footer>
