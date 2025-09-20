@php
    $customizerOptions = [
        'colorMainForeground' => [
            'title' => __('Foreground'),
            'cssVar' => '--foreground',
            'type' => 'color',
            'values' => [],
        ],
        'colorMainBackground' => [
            'title' => __('Background Color'),
            'cssVar' => '--background',
            'type' => 'color',
            'values' => [],
        ],
        'colorMainBorder' => [
            'title' => __('Border Color'),
            'cssVar' => '--border',
            'type' => 'color',
            'values' => [],
        ],
        'colorMainHeadingForeground' => [
            'title' => __('Headings Foreground'),
            'cssVar' => '--heading-foreground',
            'type' => 'color',
            'values' => [],
            'showInCustomizer' => false,
        ],
        'colorMainPrimary' => [
            'title' => __('Primary Color'),
            'cssVar' => '--primary',
            'type' => 'color',
            'values' => [],
        ],
        'colorMainPrimaryForeground' => [
            'title' => __('Primary Foreground'),
            'cssVar' => '--primary-foreground',
            'type' => 'color',
            'values' => [],
            'showInCustomizer' => false,
        ],
        'colorMainSecondary' => [
            'title' => __('Secondary Color'),
            'cssVar' => '--secondary',
            'type' => 'color',
            'values' => [],
        ],
        'colorMainSecondaryForeground' => [
            'title' => __('Secondary Foreground'),
            'cssVar' => '--secondary-foreground',
            'type' => 'color',
            'values' => [],
            'showInCustomizer' => false,
        ],
        'colorMainAccent' => [
            'title' => __('Accent Color'),
            'cssVar' => '--accent',
            'type' => 'color',
            'values' => [],
        ],
        'colorMainAccentForeground' => [
            'title' => __('Accent Foreground'),
            'cssVar' => '--accent-foreground',
            'type' => 'color',
            'values' => [],
            'showInCustomizer' => false,
        ],

        'navbarLinkBorderRadius' => [
            'title' => __('Navbar Links Radius'),
            'cssVar' => '--navbar-link-rounded',
            'type' => 'range',
            'value' => '12',
            'showInCustomizer' => false,
        ],
        'colorNavbarBackground' => [
            'title' => __('Background'),
            'cssVar' => '--navbar-background',
            'type' => 'color',
            'values' => [],
        ],
        'colorNavbarForeground' => [
            'title' => __('Foreground'),
            'cssVar' => '--navbar-foreground',
            'type' => 'color',
            'values' => [],
        ],
        'colorNavbarHoverBackground' => [
            'title' => __('Hover Background'),
            'cssVar' => '--navbar-background-hover',
            'type' => 'color',
            'values' => [],
            'showInCustomizer' => false,
        ],
        'colorNavbarHoverForeground' => [
            'title' => __('Hover Foreground'),
            'cssVar' => '--navbar-foreground-hover',
            'type' => 'color',
            'values' => [],
            'showInCustomizer' => false,
        ],
        'colorNavbarActiveBackground' => [
            'title' => __('Active Background'),
            'cssVar' => '--navbar-background-active',
            'type' => 'color',
            'values' => [],
            'showInCustomizer' => false,
        ],
        'colorNavbarActiveForeground' => [
            'title' => __('Active Foreground'),
            'cssVar' => '--navbar-foreground-active',
            'type' => 'color',
            'values' => [],
        ],

        'fontHeading' => [
            'title' => __('Heading Font'),
            'cssVar' => '--font-heading',
            'type' => 'font',
            'value' => '',
        ],
        'fontBody' => [
            'title' => __('Body Font'),
            'cssVar' => '--font-body',
            'type' => 'font',
            'value' => '',
        ],

        'buttonBorderRadius' => [
            'title' => __('Buttons Border Radius'),
            'cssVar' => '--button-rounded',
            'type' => 'range',
            'value' => '999px',
        ],
        'colorButtonBorder' => [
            'title' => __('Border'),
            'cssVar' => '--button-border',
            'type' => 'color',
            'values' => [],
        ],

        'cardBorderRadius' => [
            'title' => __('Cards Border Radius'),
            'cssVar' => '--card-rounded',
            'type' => 'range',
            'value' => '12px',
        ],
        'cardShadow' => [
            'title' => __('Cards Shadow'),
            'cssVar' => '--card-shadow',
            'type' => 'cssValues',
            'value' => '',
            'values' => [
                'none' => [
                    'label' => 'None',
                    'value' => 'none',
                ],
                'xs' => [
                    'label' => 'Extra Small',
                    'value' => '0 2px 1px var(--card-shadow-color, hsl(0 0% 0% / 5%))',
                ],
                'sm' => [
                    'label' => 'Small',
                    'value' => '0 1px 2px 0 var(--card-shadow-color, hsl(0 0% 0% / 5%))',
                ],
                'md' => [
                    'label' => 'Medium',
                    'value' => '0 4px 6px -1px var(--card-shadow-color, hsl(0 0% 0% / 5%)), 0 2px 4px -2px var(--card-shadow-color, hsl(0 0% 0% / 5%))',
                ],
                'lg' => [
                    'label' => 'Large',
                    'value' => '0 10px 15px -3px var(--card-shadow-color, hsl(0 0% 0% / 5%)), 0 4px 6px -4px var(--card-shadow-color, hsl(0 0% 0% / 5%))',
                ],
                'xl' => [
                    'label' => 'Extra Large',
                    'value' => '0 20px 25px -5px var(--card-shadow-color, hsl(0 0% 0% / 5%)), 0 8px 10px -6px var(--card-shadow-color, hsl(0 0% 0% / 5%))',
                ],
                '2xl' => [
                    'label' => '2Extra Large',
                    'value' => '0 25px 50px -12px var(--card-shadow-color, hsl(0 0% 0% / 5%))',
                ],
            ],
            'showInCustomizer' => false,
        ],
        'colorCardBackground' => [
            'title' => __('Background'),
            'cssVar' => '--card-background',
            'type' => 'color',
            'values' => [],
        ],
        'colorCardBorder' => [
            'title' => __('Border'),
            'cssVar' => '--card-border',
            'type' => 'color',
            'values' => [],
        ],

        'inputBorderRadius' => [
            'title' => __('Inputs Border Radius'),
            'cssVar' => '--input-rounded-multiplier',
            'type' => 'range',
            'value' => '1',
        ],
        'colorInputForeground' => [
            'title' => __('Foreground'),
            'cssVar' => '--input-foreground',
            'type' => 'color',
            'values' => [],
        ],
        'colorInputBackground' => [
            'title' => __('Background'),
            'cssVar' => '--input-background',
            'type' => 'color',
            'values' => [],
        ],
        'colorInputBorder' => [
            'title' => __('Border'),
            'cssVar' => '--input-border',
            'type' => 'color',
            'values' => [],
        ],
    ];

    $active_theme = setting('dash_theme') ?? 'default';

    $google_fonts = \App\Services\Common\FontsService::getGoogleFonts();
    $google_fonts_list = [];

    foreach ($google_fonts['items'] ?? [] as $font) {
        $family = $font['family'];
        $lowercase_family = strtolower($family);

        if (!str_contains($lowercase_family, 'icon') && !str_contains($lowercase_family, 'material symbols')) {
            $google_fonts_list[] = $family;
        }
    }
@endphp

<div
    class="lqd-customizer fixed bottom-8 left-1/2 z-[9999] flex -translate-x-1/2 items-center gap-2 rounded-full border border-black/10 bg-white/80 px-2 py-1.5 shadow-xl shadow-black/[7%] backdrop-blur-lg backdrop-saturate-150 transition-colors max-md:hidden"
    x-data="lqdCustomizer"
>
    @foreach ($customizerOptions as $key => $option)
        @continue($option['type'] !== 'color' || !($option['showInCustomizer'] ?? true) || strpos($key, 'colorMain') !== 0)

        @php
            $showInCustomizer = $option['showInCustomizer'] ?? true;
            $mainColorAssosiatedKeys = [];

            if ($key === 'colorMainPrimary') {
                $mainColorAssosiatedKeys = [
                    [
                        'key' => 'colorMainPrimaryForeground',
                        'manipulations' => [
                            [
                                'type' => 'autoBlackWhite',
                            ],
                        ],
                    ],
                ];
            }
            if ($key === 'colorMainSecondary') {
                $mainColorAssosiatedKeys = [
                    [
                        'key' => 'colorMainSecondaryForeground',
                        'manipulations' => [
                            [
                                'type' => 'autoBlackWhite',
                                'value' => '--secondary',
                            ],
                        ],
                    ],
                ];
            }
            if ($key === 'colorMainAccent') {
                $mainColorAssosiatedKeys = [
                    [
                        'key' => 'colorMainAccentForeground',
                        'manipulations' => [
                            [
                                'type' => 'autoBlackWhite',
                                'value' => '--accent',
                            ],
                        ],
                    ],
                ];
            }
            if ($key === 'colorMainForeground') {
                $mainColorAssosiatedKeys = [
                    [
                        'key' => 'colorMainHeadingForeground',
                        'manipulations' => [
                            [
                                'type' => 'darken',
                                'value' => 22,
                                'condition' => 'lightMode',
                            ],
                            [
                                'type' => 'lighten',
                                'value' => 30,
                                'condition' => 'darkMode',
                            ],
                            [
                                'type' => 'desaturate',
                                'value' => 7,
                                'condition' => 'lightMode',
                            ],
                            [
                                'type' => 'desaturate',
                                'value' => 30,
                                'condition' => 'darkMode',
                            ],
                        ],
                    ],
                ];
            }
        @endphp

        <label
            class="relative inline-block size-9 shrink-0 grow-0 cursor-pointer rounded-full border border-black/10 transition-all before:pointer-events-none before:invisible before:absolute before:bottom-full before:start-1/2 before:mb-2 before:-translate-x-1/2 before:translate-y-1 before:whitespace-nowrap before:rounded-full before:border before:border-black/10 before:bg-white/90 before:px-2 before:py-0.5 before:text-[12px] before:font-medium before:text-black/90 before:opacity-0 before:transition-all before:content-[attr(title)] hover:scale-105 hover:before:visible hover:before:translate-y-0 hover:before:opacity-100"
            type="button"
            title="{{ $option['title'] }}"
            style="background-color: hsl(var({{ $option['cssVar'] }}));"
            for="lqd-customizer-{{ $key }}"
        >
            <input
                class="absolute inset-0 size-full cursor-pointer opacity-0"
                id="lqd-customizer-{{ $key }}"
                type="color"
                @input="onColorInput({key: '{{ $key }}', color: $event.target.value, assosiatedKeys: {{ json_encode($mainColorAssosiatedKeys) }}})"
                :value="options.{{ $key }}.values?.[darkMode ? 'dark' : 'light']?.hex || parseColor({ color: options.{{ $key }}.cssVar, format: 'hex' })"
                value="#000000"
            >
        </label>
    @endforeach

    <x-button
        class="size-9 shrink-0 grow-0 rounded-full border border-black/10 text-black/90 before:pointer-events-none before:invisible before:absolute before:bottom-full before:start-1/2 before:mb-2 before:-translate-x-1/2 before:translate-y-1 before:whitespace-nowrap before:rounded-full before:border before:border-black/10 before:bg-white/90 before:px-2 before:py-0.5 before:text-[12px] before:font-medium before:text-black/90 before:opacity-0 before:transition-all before:content-[attr(title)] hover:scale-105 hover:before:visible hover:before:translate-y-0 hover:before:opacity-100"
        variant="none"
        size="none"
        title="{{ __('Random Colors') }}"
        href="#"
        @click.prevent="randomColors"
    >
        <x-tabler-arrows-shuffle class="size-4" />
    </x-button>

    <span class="mx-1 inline-block h-6 w-px bg-black/10 transition-colors"></span>

    <x-modal
        class:modal="bottom-full overflow-visible end-0 start-auto min-w-[320px] max-w-[90vw] top-auto mb-3 mt-0"
        class:modal-backdrop="invisible"
        class:modal-content="max-h-[calc(100vh-110px)] overflow-visible border border-black/5 bg-white shadow-xl shadow-black/5"
        class:modal-body="p-3"
        type="inline"
    >
        <x-slot:trigger
            class="size-9 shrink-0 grow-0 rounded-full border border-black/10 text-black/90 before:pointer-events-none before:invisible before:absolute before:bottom-full before:start-1/2 before:mb-2 before:-translate-x-1/2 before:translate-y-1 before:whitespace-nowrap before:rounded-full before:border before:border-black/10 before:bg-white/90 before:px-2 before:py-0.5 before:text-[12px] before:font-medium before:text-black/90 before:opacity-0 before:transition-all before:content-[attr(title)] hover:scale-105 hover:before:visible hover:before:translate-y-0 hover:before:opacity-100"
            variant="none"
            size="none"
            title="{{ __('Typography') }}"
            href="#"
        >
            <x-tabler-letter-case class="size-4" />
        </x-slot:trigger>

        <x-slot:modal>
            <div class="flex flex-col gap-4">

                @foreach ($customizerOptions as $key => $option)
                    @continue($option['type'] !== 'font')

                    <div x-data="lqdCustomizerFontPicker('{{ $key }}')">
                        <p class="text-[12px] font-medium text-black/60">
                            {{ $option['title'] }}
                        </p>

                        <div class="relative">
                            <button
                                class="flex w-full cursor-pointer items-center justify-between gap-1 rounded-lg border border-black/5 p-2 text-2xs text-black/90 transition-all hover:bg-black/[2%]"
                                type="button"
                                @click="dropdownOpen = !dropdownOpen"
                                :style="{ 'fontFamily': selectedFont + ', sans-serif' }"
                            >
                                <span x-text="selectedFont || '{{ __('Choose a Font') }}'"></span>

                                <x-tabler-chevron-down
                                    class="size-4 transition-transform"
                                    ::class="{ 'rotate-180': dropdownOpen }"
                                />
                            </button>

                            <div
                                class="absolute -inset-x-1 bottom-full z-2 mb-2 rounded-lg border border-black/5 bg-white pb-3 text-xs shadow-lg shadow-black/5"
                                x-cloak
                                x-show="dropdownOpen"
                                @click.outside="dropdownOpen = false"
                                x-transition
                                @keydown.escape.window="dropdownOpen = false"
                            >
                                <form class="relative p-3">
                                    <x-tabler-search class="absolute start-6 top-1/2 size-4 -translate-y-1/2" />
                                    <input
                                        class="w-full rounded-md border border-black/5 py-2 pe-2 ps-8 text-2xs"
                                        type="search"
                                        placeholder="{{ __('Search for a font') }}"
                                        @input.debounce.50ms="$event.target.value.length >= 3 && onSearchInput()"
                                        x-model="searchString"
                                        x-trap="dropdownOpen"
                                    />
                                </form>

                                <ul class="max-h-80 overflow-y-auto overscroll-contain px-3">
                                    <template
                                        x-for="(font, index) in showingGoogleFonts"
                                        :key="font"
                                    >
                                        <li
                                            class="cursor-pointer rounded-md p-2 text-2xs font-medium text-black/90 transition-all hover:bg-black/[2%]"
                                            @click="addToGoogleFontLoadQueue(font); selectedFont = font; dropdownOpen = false"
                                            x-init="index <= 10 && addToGoogleFontLoadQueue(font, true)"
                                            x-intersect:enter.once="addToGoogleFontLoadQueue(font, true)"
                                            :style="{ 'fontFamily': font + ', sans-serif' }"
                                        >
                                            <span x-text="font"></span>
                                        </li>
                                    </template>
                                    <li
                                        class="min-h-px opacity-0"
                                        x-show="!searchString.trim()"
                                        x-intersect="loadMoreFonts"
                                    ></li>
                                    <li
                                        class="cursor-default rounded-md p-2 text-2xs font-medium text-black/60 transition-all"
                                        x-show="showingGoogleFonts.length === 0"
                                    >
                                        {{ __('No fonts found') }}
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                @endforeach
            </div>
        </x-slot:modal>
    </x-modal>

    <span class="mx-1 inline-block h-6 w-px bg-black/10 transition-colors"></span>

    <x-modal
        class:modal="bottom-full overflow-visible end-0 start-auto min-w-[320px] max-w-[90vw] top-auto mb-3 mt-0"
        class:modal-content="max-h-[calc(100vh-110px)] overflow-visible border border-black/5 bg-white shadow-xl shadow-black/5"
        class:modal-backdrop="invisible"
        class:modal-body="p-3"
        type="inline"
    >
        <x-slot:trigger
            class="size-9 shrink-0 grow-0 rounded-full border border-black/10 text-black/90 before:pointer-events-none before:invisible before:absolute before:bottom-full before:start-1/2 before:mb-2 before:-translate-x-1/2 before:translate-y-1 before:whitespace-nowrap before:rounded-full before:border before:border-black/10 before:bg-white/90 before:px-2 before:py-0.5 before:text-[12px] before:font-medium before:text-black/90 before:opacity-0 before:transition-all before:content-[attr(title)] hover:scale-105 hover:before:visible hover:before:translate-y-0 hover:before:opacity-100"
            variant="none"
            size="none"
            title="{{ __('Navbar') }}"
            href="#"
        >
            <x-tabler-layout-sidebar class="size-4 rtl:rotate-180" />
        </x-slot:trigger>

        <x-slot:modal>
            <div>
                @if ($active_theme !== 'bolt' && $active_theme !== 'social-media' && $active_theme !== 'social-media-dashboard')
                    <x-forms.input
                        class:label="flex-row-reverse justify-between text-black/90"
                        class="border-black/10"
                        type="checkbox"
                        size="sm"
                        switcher
                        label="{{ __('Collapse Navbar') }}"
                        @change="$store.navbarShrink.toggle()"
                        ::checked="$store.navbarShrink.active"
                    />

                    <hr class="!my-3 border-black/10" />
                @endif

                <div class="flex w-full flex-col text-black dark:text-black">
                    <label
                        class="text-[12px] font-medium text-black/90"
                        for="navbar-link-border-radius"
                    >
                        {{ __('Links Border Radius') }}
                    </label>
                    <input
                        class="z-10 mb-2 mt-2 h-1 w-full appearance-none rounded-full bg-black/10 focus:outline-primary [&::-moz-range-thumb]:size-3 [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-primary active:[&::-moz-range-thumb]:scale-110 [&::-webkit-slider-thumb]:size-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:bg-primary active:[&::-webkit-slider-thumb]:scale-110"
                        id="navbar-link-border-radius"
                        type="range"
                        min="0"
                        max="30"
                        :value="parseInt(options.navbarLinkBorderRadius.value)"
                        @input="options.navbarLinkBorderRadius.value = $event.target.value + 'px'"
                    />
                </div>

                <hr class="border-black/10" />

                <div class="space-y-2">
                    @foreach ($customizerOptions as $key => $option)
                        @continue($option['type'] !== 'color' || !($option['showInCustomizer'] ?? true) || strpos($key, 'colorNavbar') !== 0)

                        @php
                            $navbarColorAssosiatedKeys = [];

                            if ($key === 'colorNavbarActiveForeground') {
                                $navbarColorAssosiatedKeys = [
                                    [
                                        'key' => 'colorNavbarActiveBackground',
                                    ],
                                    [
                                        'key' => 'colorNavbarHoverForeground',
                                    ],
                                    [
                                        'key' => 'colorNavbarHoverBackground',
                                    ],
                                ];
                            }
                        @endphp

                        <label
                            class="flex cursor-pointer items-center gap-2 text-2xs font-medium text-black/90"
                            type="button"
                            for="lqd-customizer-{{ $key }}"
                        >
                            <span
                                class="relative inline-block size-7 shrink-0 grow-0 rounded-full border border-black/10 transition-all"
                                style="background-color: hsl(var({{ $option['cssVar'] }}));"
                            >
                                <input
                                    class="absolute inset-0 size-full cursor-pointer opacity-0"
                                    id="lqd-customizer-{{ $key }}"
                                    type="color"
                                    @input="onColorInput({key: '{{ $key }}', color: $event.target.value, assosiatedKeys: {{ json_encode($navbarColorAssosiatedKeys) }}})"
                                    :value="options.{{ $key }}.values?.[darkMode ? 'dark' : 'light']?.hex || parseColor({
                                        color: options.{{ $key }}.cssVar,
                                        format: 'hex'
                                    })")"
                                    value="#000000"
                                >
                            </span>
                            {{ $option['title'] }}
                        </label>
                    @endforeach
                </div>
            </div>
        </x-slot:modal>
    </x-modal>

    <span class="mx-1 inline-block h-6 w-px bg-black/10 transition-colors"></span>

    <x-modal
        class:modal="bottom-full overflow-visible end-0 start-auto min-w-[320px] max-w-[90vw] top-auto mb-3 mt-0"
        class:modal-content="max-h-[calc(100vh-110px)] overflow-y-auto border border-black/5 bg-white shadow-xl shadow-black/5"
        class:modal-backdrop="invisible"
        class:modal-body="p-3"
        type="inline"
    >
        <x-slot:trigger
            class="size-9 shrink-0 grow-0 rounded-full border border-black/10 text-black/90 before:pointer-events-none before:invisible before:absolute before:bottom-full before:start-1/2 before:mb-2 before:-translate-x-1/2 before:translate-y-1 before:whitespace-nowrap before:rounded-full before:border before:border-black/10 before:bg-white/90 before:px-2 before:py-0.5 before:text-[12px] before:font-medium before:text-black/90 before:opacity-0 before:transition-all before:content-[attr(title)] hover:scale-105 hover:before:visible hover:before:translate-y-0 hover:before:opacity-100"
            variant="none"
            size="none"
            title="{{ __('Components') }}"
            href="#"
        >
            <x-tabler-triangle-square-circle class="size-4" />
        </x-slot:trigger>

        <x-slot:modal>
            <div class="space-y-5 py-4">
                <div class="rounded-lg border border-black/10 p-4">
                    <h4 class="-mt-7 mb-5 text-black">
                        <span class="-ms-1 inline-block bg-white pe-3 ps-1">
                            {{ __('Buttons') }}
                        </span>
                    </h4>

                    <div class="space-y-2">
                        <div class="flex w-full flex-col text-black dark:text-black">
                            <label
                                class="text-[12px] font-medium text-black/90"
                                for="buttons-border-radius"
                            >
                                {{ __('Border Radius') }}
                            </label>
                            <input
                                class="z-10 mb-2 mt-2 h-1 w-full appearance-none rounded-full bg-black/10 focus:outline-primary [&::-moz-range-thumb]:size-3 [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-primary active:[&::-moz-range-thumb]:scale-110 [&::-webkit-slider-thumb]:size-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:bg-primary active:[&::-webkit-slider-thumb]:scale-110"
                                id="buttons-border-radius"
                                type="range"
                                min="0"
                                max="30"
                                :value="parseInt(options.buttonBorderRadius.value)"
                                @input="options.buttonBorderRadius.value = $event.target.value + 'px'"
                            />
                        </div>

                        <hr class="!my-3 border-black/10" />

                        @foreach ($customizerOptions as $key => $option)
                            @continue($option['type'] !== 'color' || !($option['showInCustomizer'] ?? true) || strpos($key, 'colorButton') !== 0)

                            <label
                                class="flex cursor-pointer items-center gap-2 text-2xs font-medium text-black/90"
                                type="button"
                                for="lqd-customizer-{{ $key }}"
                            >
                                <span
                                    class="relative inline-block size-7 shrink-0 grow-0 rounded-full border border-black/10 transition-all"
                                    style="background-color: hsl(var({{ $option['cssVar'] }}));"
                                >
                                    <input
                                        class="absolute inset-0 size-full cursor-pointer opacity-0"
                                        id="lqd-customizer-{{ $key }}"
                                        type="color"
                                        @input="onColorInput({key: '{{ $key }}', color: $event.target.value })"
                                        :value="options.{{ $key }}.values?.[darkMode ? 'dark' : 'light']?.hex || parseColor({
                                            color: options.{{ $key }}.cssVar,
                                            format: 'hex'
                                        })")"
                                        value="#000000"
                                    >
                                </span>
                                {{ $option['title'] }}
                            </label>
                        @endforeach
                    </div>
                </div>

                <div class="rounded-lg border border-black/10 p-4">
                    <h4 class="-mt-7 mb-5 text-black">
                        <span class="-ms-1 inline-block bg-white pe-3 ps-1">
                            {{ __('Cards') }}
                        </span>
                    </h4>

                    <div class="space-y-2">
                        <div class="flex w-full flex-col text-black dark:text-black">
                            <label
                                class="text-[12px] font-medium text-black/90"
                                for="cards-border-radius"
                            >
                                {{ __('Border Radius') }}
                            </label>
                            <input
                                class="z-10 mb-2 mt-2 h-1 w-full appearance-none rounded-full bg-black/10 focus:outline-primary [&::-moz-range-thumb]:size-3 [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-primary active:[&::-moz-range-thumb]:scale-110 [&::-webkit-slider-thumb]:size-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:bg-primary active:[&::-webkit-slider-thumb]:scale-110"
                                id="cards-border-radius"
                                type="range"
                                value="20"
                                min="0"
                                max="30"
                                :value="parseInt(options.cardBorderRadius.value)"
                                @input="options.cardBorderRadius.value = $event.target.value + 'px'"
                            />
                        </div>

                        <div class="flex items-center justify-between gap-2">
                            <label class="text-[12px] font-medium text-black/90">
                                {{ __('Shadow') }}
                            </label>
                            <div class="flex items-center gap-1">
                                @foreach ($customizerOptions['cardShadow']['values'] as $key => $shadow)
                                    <label
                                        class="relative inline-grid size-7 cursor-pointer place-content-center place-items-center rounded border border-black/10 bg-white text-black/90 shadow-card transition-all [&.active]:bg-black/90 [&.active]:text-white"
                                        for="lqd-customizer-shadow-{{ $key }}-card"
                                        style="--card-shadow-color: hsl(0 0% 0% / 20%); --card-shadow: {{ $key === '2xl' ? '0 20px 30px 10px var(--card-shadow-color, hsl(0 0% 0% / 50%))' : $shadow['value'] }}; z-index: {{ $loop->count - $loop->index }};"
                                        :class="{ 'active': options.cardShadow.value === '{{ $shadow['value'] }}' }"
                                    >
                                        <input
                                            class="absolute size-full cursor-pointer opacity-0"
                                            id="lqd-customizer-shadow-{{ $key }}-card"
                                            x-model="options.cardShadow.value"
                                            type="radio"
                                            value="{{ $shadow['value'] }}"
                                        />
                                        <span class="w-full overflow-hidden text-ellipsis whitespace-nowrap text-center text-[10px] font-semibold uppercase tracking-wide">
                                            {{ $key }}
                                        </span>
                                    </label>
                                @endforeach
                            </div>
                        </div>

                        <hr class="!my-3 border-black/10" />

                        @foreach ($customizerOptions as $key => $option)
                            @continue($option['type'] !== 'color' || !($option['showInCustomizer'] ?? true) || strpos($key, 'colorCard') !== 0)

                            <label
                                class="flex cursor-pointer items-center gap-2 text-2xs font-medium text-black/90"
                                type="button"
                                for="lqd-customizer-{{ $key }}"
                            >
                                <span
                                    class="relative inline-block size-7 shrink-0 grow-0 rounded-full border border-black/10 transition-all"
                                    style="background-color: hsl(var({{ $option['cssVar'] }}));"
                                >
                                    <input
                                        class="absolute inset-0 size-full cursor-pointer opacity-0"
                                        id="lqd-customizer-{{ $key }}"
                                        type="color"
                                        @input="onColorInput({key: '{{ $key }}', color: $event.target.value })"
                                        :value="options.{{ $key }}.values?.[darkMode ? 'dark' : 'light']?.hex || parseColor({
                                            color: options.{{ $key }}.cssVar,
                                            format: 'hex'
                                        })")"
                                        value="#000000"
                                    >
                                </span>
                                {{ $option['title'] }}
                            </label>
                        @endforeach
                    </div>
                </div>

                <div class="rounded-lg border border-black/10 p-4">
                    <h4 class="-mt-7 mb-5 text-black">
                        <span class="-ms-1 inline-block bg-white pe-3 ps-1">
                            {{ __('Inputs') }}
                        </span>
                    </h4>

                    <div class="space-y-2">
                        <div class="flex w-full flex-col text-black dark:text-black">
                            <label
                                class="text-[12px] font-medium text-black/90"
                                for="inputs-border-radius"
                            >
                                {{ __('Border Radius') }}
                            </label>
                            <input
                                class="z-10 mb-2 mt-2 h-1 w-full appearance-none rounded-full bg-black/10 focus:outline-primary [&::-moz-range-thumb]:size-3 [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-primary active:[&::-moz-range-thumb]:scale-110 [&::-webkit-slider-thumb]:size-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:bg-primary active:[&::-webkit-slider-thumb]:scale-110"
                                id="inputs-border-radius"
                                type="range"
                                value="1"
                                min="0"
                                max="3"
                                step="0.2"
                                x-model="options.inputBorderRadius.value"
                            />
                        </div>

                        <hr class="!my-3 border-black/10" />

                        @foreach ($customizerOptions as $key => $option)
                            @continue($option['type'] !== 'color' || !($option['showInCustomizer'] ?? true) || strpos($key, 'colorInput') !== 0)

                            <label
                                class="flex cursor-pointer items-center gap-2 text-2xs font-medium text-black/90"
                                type="button"
                                for="lqd-customizer-{{ $key }}"
                            >
                                <span
                                    class="relative inline-block size-7 shrink-0 grow-0 rounded-full border border-black/10 transition-all"
                                    style="background-color: hsl(var({{ $option['cssVar'] }}));"
                                >
                                    <input
                                        class="absolute inset-0 size-full cursor-pointer opacity-0"
                                        id="lqd-customizer-{{ $key }}"
                                        type="color"
                                        @input="onColorInput({key: '{{ $key }}', color: $event.target.value })"
                                        :value="options.{{ $key }}.values?.[darkMode ? 'dark' : 'light']?.hex || parseColor({
                                            color: options.{{ $key }}.cssVar,
                                            format: 'hex'
                                        })")"
                                        value="#000000"
                                    >
                                </span>
                                {{ $option['title'] }}
                            </label>
                        @endforeach
                    </div>
                </div>
            </div>
        </x-slot:modal>
    </x-modal>

    <span class="mx-1 inline-block h-6 w-px bg-black/10 transition-colors"></span>

    <x-light-dark-switch class="size-9 shrink-0 grow-0 rounded-full border border-black/10 text-black/90 hover:translate-y-0" />

    <span class="mx-1 inline-block h-6 w-px bg-black/10 transition-colors"></span>

    <x-modal
        class:modal="bottom-full overflow-visible end-0 start-auto top-auto mb-3 mt-0 w-[min(400px,100%)]"
        class:modal-backdrop="invisible"
        class:modal-content="max-h-[calc(100vh-110px)] overflow-visible border border-black/5 bg-white shadow-xl shadow-black/5"
        class:modal-body="p-3"
        type="inline"
    >
        <x-slot:trigger
            class="size-9 shrink-0 grow-0 rounded-full border border-black/10 text-black/90 before:pointer-events-none before:invisible before:absolute before:bottom-full before:start-1/2 before:mb-2 before:-translate-x-1/2 before:translate-y-1 before:whitespace-nowrap before:rounded-full before:border before:border-black/10 before:bg-white/90 before:px-2 before:py-0.5 before:text-[12px] before:font-medium before:text-black/90 before:opacity-0 before:transition-all before:content-[attr(title)] hover:scale-105 hover:before:visible hover:before:translate-y-0 hover:before:opacity-100"
            variant="none"
            size="none"
            title="{{ __('Export Styles') }}"
            href="#"
        >
            <x-tabler-download class="size-4" />
        </x-slot:trigger>

        <x-slot:modal>
            <div class="relative rounded-lg border border-black/5 bg-slate-50 text-black/90">
                <p
                    class="m-0 p-4 text-2xs font-medium"
                    x-show="!styleString.length"
                >
                    {{ __('Start Customizing colors and other properties.') }}
                </p>
                <div
                    class="relative w-full"
                    x-show="styleString.length"
                >
                    <pre class="max-h-[min(440px,75vh)] w-full overflow-y-auto px-5 py-4"><code class="w-full text-2xs/6 font-medium" x-text="styleString"></code></pre>

                    <div class="absolute end-3 top-3 z-2 flex gap-3">
                        <x-button
                            class="size-9 shrink-0 rounded-full border border-black/10 text-black/90 backdrop-blur hover:scale-105"
                            variant="none"
                            size="none"
                            title="{{ __('Copy') }}"
                            @click.prevent="navigator.clipboard.writeText(styleString); toastr.success('{{ __('Copied to clipboard') }}')"
                        >
                            <x-tabler-copy class="size-4" />
                        </x-button>
                    </div>
                </div>
            </div>
        </x-slot:modal>
    </x-modal>

    <x-modal
        class:modal="bottom-full overflow-visible end-0 start-auto top-auto mb-3 mt-0"
        class:modal-backdrop="invisible"
        class:modal-content="max-h-[calc(100vh-110px)] overflow-visible border border-black/5 bg-white shadow-xl shadow-black/5"
        class:modal-body="p-3"
        type="inline"
    >
        <x-slot:trigger
            class="size-9 shrink-0 grow-0 rounded-full border border-black/10 text-black/90 before:pointer-events-none before:invisible before:absolute before:bottom-full before:start-1/2 before:mb-2 before:-translate-x-1/2 before:translate-y-1 before:whitespace-nowrap before:rounded-full before:border before:border-black/10 before:bg-white/90 before:px-2 before:py-0.5 before:text-[12px] before:font-medium before:text-black/90 before:opacity-0 before:transition-all before:content-[attr(title)] hover:scale-105 hover:before:visible hover:before:translate-y-0 hover:before:opacity-100"
            variant="none"
            size="none"
            title="{{ __('Import Styles') }}"
            href="#"
        >
            <x-tabler-upload class="size-4" />
        </x-slot:trigger>

        <x-slot:modal>
            <textarea
                class="w-full rounded-lg border border-black/5 bg-slate-50 px-5 py-4 font-mono text-2xs/6 font-medium text-black/90"
                rows="12"
                placeholder="{{ __('Paste your styles here') }}"
                @input.throttle.30ms="onImportStyles($event.target.value)"
            ></textarea>
        </x-slot:modal>
    </x-modal>

    <span class="mx-1 inline-block h-6 w-px bg-black/10 transition-colors"></span>

    <x-button
        class="size-9 shrink-0 grow-0 rounded-full border border-black/10 text-black/90 before:pointer-events-none before:invisible before:absolute before:bottom-full before:start-1/2 before:mb-2 before:-translate-x-1/2 before:translate-y-1 before:whitespace-nowrap before:rounded-full before:border before:border-black/10 before:bg-white/90 before:px-2 before:py-0.5 before:text-[12px] before:font-medium before:text-black/90 before:opacity-0 before:transition-all before:content-[attr(title)] hover:scale-105 hover:bg-orange-400 hover:text-white hover:before:visible hover:before:translate-y-0 hover:before:opacity-100"
        variant="none"
        size="none"
        title="{{ __('Reset Styles') }}"
        href="#"
        x-cloak
        x-show="styleString.replace(/[\n\t]/g, '').trim().length"
        @click.prevent="resetStyles"
    >
        <x-tabler-refresh class="size-4" />
    </x-button>

    <x-button
        class="size-9 shrink-0 grow-0 rounded-full border border-black/10 text-black/90 before:pointer-events-none before:invisible before:absolute before:bottom-full before:start-1/2 before:mb-2 before:-translate-x-1/2 before:translate-y-1 before:whitespace-nowrap before:rounded-full before:border before:border-black/10 before:bg-white/90 before:px-2 before:py-0.5 before:text-[12px] before:font-medium before:text-black/90 before:opacity-0 before:transition-all before:content-[attr(title)] hover:scale-105 hover:bg-lime-600 hover:text-white hover:before:visible hover:before:translate-y-0 hover:before:opacity-100"
        variant="none"
        size="none"
        title="{{ __('Save & Close Customizer') }}"
        href="#"
        @click.prevent="saveAndClose"
    >
        <x-tabler-check class="size-4" />
    </x-button>

    <div class="relative">
        <x-button
            class="size-9 shrink-0 grow-0 rounded-full border border-black/10 text-black/90 before:pointer-events-none before:invisible before:absolute before:bottom-full before:start-1/2 before:mb-2 before:-translate-x-1/2 before:translate-y-1 before:whitespace-nowrap before:rounded-full before:border before:border-black/10 before:bg-white/90 before:px-2 before:py-0.5 before:text-[12px] before:font-medium before:text-black/90 before:opacity-0 before:transition-all before:content-[attr(title)] hover:scale-105 hover:bg-red-500 hover:text-white hover:before:visible hover:before:translate-y-0 hover:before:opacity-100"
            variant="none"
            size="none"
            title="{{ __('Discard Changes') }}"
            href="#"
            @click.prevent="discardChanges"
        >
            <x-tabler-x class="size-4" />
        </x-button>

        <x-info-tooltip
            class="absolute -end-2 -top-2 z-2"
            class:icon="[&_svg]:size-5 text-orange-500 opacity-100"
            text="{{ __('Your changes are not saved and will be lost if you close the customizer.') }}"
            x-show="styleString.replace(/[\n\t]/g, '').trim().length"
            x-cloak
        />
    </div>
</div>

@push('script')
    <script>
        window.lqdGoogleFontsList = @json($google_fonts_list);
        window.lqdCustomizerOptions = @json($customizerOptions);
    </script>
    <script src="{{ custom_theme_url('/assets/libs/tinycolor/tinycolor-min.js') }}"></script>
@endpush
