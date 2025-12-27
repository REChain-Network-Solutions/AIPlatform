@php
    $sales_prev_week = cache('sales_previous_week');
    $sales_this_week = cache('sales_this_week');

    $api_cost_distribution = cache('api_cost_distribution');
    $popular_plans_data = cache('popular_plans_data');
    $popular_tools_data = cache('popular_tools_data');
    $currencySymbol = currency()->symbol;

    $premium_features = [
        'VIP Support' => 'Get instant help whenever you need it.',
        'Access to All Current & Future Extensions <span class="font-bold text-[#6977DE]">worth $2000+</span>' => 'Always stay ahead with the latest features.' ,
        'Access to All Current & Future Themes <span class="font-bold text-[#6977DE]">worth $670</span>'=> 'Always stay ahead with the latest designs.' ,
		'Get the Mobile App Free in Your 4th Month! <span class="font-bold text-[#6977DE]">worth $3000+</span>' => 'Enjoy a free mobile app after your fourth month of subscription.' ,
        '10 Hours of Custom Development Every Month' => 'Tailored improvements, at no extra cost.' ,
        'Direct Communication with Our Development Team' => 'No middlemen, just solutions.' ,
        'Exclusive Extensions Not Available to Others' => 'Stay ahead of competition, reserved for VIPs only.' ,
        'Complimentary Logo Design' => 'A custom logo to elevate your brand.' ,
        'Personalized Onboarding Assistance' => 'We’ll help you get up and running smoothly.' ,
        'Free Setup & Configuration Services' => 'Let us handle the technical details for you.' ,
    ];
@endphp

@extends('panel.layout.app', ['disable_tblr' => true, 'disable_titlebar' => true])
@section('title', __('Overview'))

@section('content')
    <div class="py-10">
		@if($vip_membership && data_get($vip_membership, 'stripe_status') === 'past_due' && data_get($vip_membership, 'updateLink'))
			<x-alert variant="danger" class="mb-6" size="md">
				<p>
					{{ data_get($vip_membership, 'text') }}
				</p>
				<a
					target="_blank"
					class="text-blue-600"
					href="{{ data_get($vip_membership, 'updateLink') }}"
				>@lang('Update Payment Information')</a>
			</x-alert>
		@endif

        @if ($gatewayError == true)
            <x-alert class="mb-11">
                <p>
                    {{ __('Gateway is set to use sandbox. Please set mode to development!') }}<br><br>
                </p>
                <ul class="flex list-inside list-disc flex-col gap-3 [&_ol]:mt-2 [&_ol]:flex [&_ol]:list-inside [&_ol]:list-decimal [&_ol]:flex-col [&_ol]:gap-1 [&_ol]:ps-4">
                    <li>
                        {{ __('To use live settings:') }}
                        <ol>
                            <li>{{ __('Set mode to Production') }}</li>
                            <li>{{ __('Save gateway settings') }}</li>
                            <li>{{ __('Know that all defined products and prices will reset.') }}</li>
                        </ol>
                    </li>
                    <li>
                        {{ __('To use sandbox settings:') }}
                        <ol>
                            <li>{{ __('Set mode to Development') }}</li>
                            <li>{{ __('Save gateway settings') }}</li>
                            <li>{{ __('Know that all defined products and prices will reset.') }}</li>
                        </ol>
                    </li>
                    <li>{{ __('Beware of that order is important. First set mode then save gateway settings.') }}</li>
                </ul>
            </x-alert>
        @endif

        {{-- beging: brand-header-caption --}}
        <div class="mb-4 flex items-center justify-between">
            <h2 class="font-bold">Hey {{ auth()?->user()?->name }} 👋</h2>
            <x-modal
                class:modal-body="p-8"
                title="{{ __('Customize the widgets') }}"
                disable-modal="{{ $app_is_demo }}"
                disable-modal-message="{{ __('This feature is disabled in Demo version.') }}"
            >
                <x-slot:trigger
                    custom
                >
                    <x-button
                        variant="ghost-shadow"
                        @click.prevent="toggleModal()"
                    >
                        {{ __('Customize') }}
                    </x-button>
                </x-slot:trigger>
                <x-slot:modal>
                    <div class="lqd-user-menu-list">
                        <ol class="lqd-menu-list flex flex-col gap-2">
                            @foreach (cache('dashboard_widgets', []) as $widget)
								@if($widget->name === \App\Enums\DashboardWidget::PREMIUM_ADVANTAGES)
									@continue
								@endif

                                <li
                                    class="group/item text-xs font-medium"
                                    id="{{ 'menu-' . $widget['id'] }}"
                                    data-name="{{ $widget->name->value }}"
                                >
                                    <div class="items-center gap-5 rounded-xl border bg-background px-4 py-3 transition-all hover:shadow-lg hover:shadow-black/5">
                                        <div class="flex w-full items-center gap-5">
                                            <div class="lqd-menu-item-handle flex size-6 cursor-grab items-center justify-center">
                                                <svg
                                                    width="10"
                                                    height="16"
                                                    viewBox="0 0 10 16"
                                                    fill="none"
                                                    xmlns="http://www.w3.org/2000/svg"
                                                >
                                                    <path
                                                        d="M2 16C1.45 16 0.979167 15.8042 0.5875 15.4125C0.195833 15.0208 0 14.55 0 14C0 13.45 0.195833 12.9792 0.5875 12.5875C0.979167 12.1958 1.45 12 2 12C2.55 12 3.02083 12.1958 3.4125 12.5875C3.80417 12.9792 4 13.45 4 14C4 14.55 3.80417 15.0208 3.4125 15.4125C3.02083 15.8042 2.55 16 2 16ZM8 16C7.45 16 6.97917 15.8042 6.5875 15.4125C6.19583 15.0208 6 14.55 6 14C6 13.45 6.19583 12.9792 6.5875 12.5875C6.97917 12.1958 7.45 12 8 12C8.55 12 9.02083 12.1958 9.4125 12.5875C9.80417 12.9792 10 13.45 10 14C10 14.55 9.80417 15.0208 9.4125 15.4125C9.02083 15.8042 8.55 16 8 16ZM2 10C1.45 10 0.979167 9.80417 0.5875 9.4125C0.195833 9.02083 0 8.55 0 8C0 7.45 0.195833 6.97917 0.5875 6.5875C0.979167 6.19583 1.45 6 2 6C2.55 6 3.02083 6.19583 3.4125 6.5875C3.80417 6.97917 4 7.45 4 8C4 8.55 3.80417 9.02083 3.4125 9.4125C3.02083 9.80417 2.55 10 2 10ZM8 10C7.45 10 6.97917 9.80417 6.5875 9.4125C6.19583 9.02083 6 8.55 6 8C6 7.45 6.19583 6.97917 6.5875 6.5875C6.97917 6.19583 7.45 6 8 6C8.55 6 9.02083 6.19583 9.4125 6.5875C9.80417 6.97917 10 7.45 10 8C10 8.55 9.80417 9.02083 9.4125 9.4125C9.02083 9.80417 8.55 10 8 10ZM2 4C1.45 4 0.979167 3.80417 0.5875 3.4125C0.195833 3.02083 0 2.55 0 2C0 1.45 0.195833 0.979167 0.5875 0.5875C0.979167 0.195833 1.45 0 2 0C2.55 0 3.02083 0.195833 3.4125 0.5875C3.80417 0.979167 4 1.45 4 2C4 2.55 3.80417 3.02083 3.4125 3.4125C3.02083 3.80417 2.55 4 2 4ZM8 4C7.45 4 6.97917 3.80417 6.5875 3.4125C6.19583 3.02083 6 2.55 6 2C6 1.45 6.19583 0.979167 6.5875 0.5875C6.97917 0.195833 7.45 0 8 0C8.55 0 9.02083 0.195833 9.4125 0.5875C9.80417 0.979167 10 1.45 10 2C10 2.55 9.80417 3.02083 9.4125 3.4125C9.02083 3.80417 8.55 4 8 4Z"
                                                        fill="#A6A5AB"
                                                    />
                                                </svg>
                                            </div>

                                            <div class="flex grow items-center gap-3">
                                                <span class="inline-flex shrink-0 items-center justify-center">
                                                    {{ $widget->name->label() }}
                                                </span>
                                            </div>

                                            <div class="ms-auto flex items-center gap-2">
                                                <x-forms.input
                                                    class="h-4 w-8 [background-size:10px]"
                                                    data-href="{{ route('dashboard.admin.dashboard-widget.status', $widget['id']) }}"
                                                    data-status="menu"
                                                    data-name="{{ $widget->name->value }}"
                                                    type="checkbox"
                                                    switcher
                                                    :checked="$widget['enabled'] == '1'"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                </li>
                            @endforeach
                        </ol>
                    </div>
                </x-slot:modal>
            </x-modal>
        </div>
        {{-- end: brandh-header-caption --}}

        <div class="flex flex-col gap-11">
            {{-- begin: brand --}}
            <x-card
                class="overflow-hidden px-2 py-4 hover:-translate-y-1 hover:bg-foreground/5"
                variant="outline"
                size="lg"
            >
                <div class="relative z-1 w-full lg:w-1/2">
                    <h2 class="mb-2.5 font-bold">
                        @lang('Marketplace is here.')
                    </h2>
                    <p class="mb-0 text-sm font-medium text-foreground/80">
                        @lang('Extend the capabilities of MagicAI, explore new designs and unlock new horizons.')
                    </p>
                </div>
                <figure
                    class="absolute end-0 top-full max-w-md max-lg:-translate-y-16 lg:-end-24 lg:-top-16"
                    aria-hidden="true"
                >
                    <img
                        class="w-full"
                        alt="{{ __('marketplace') }}"
                        width="857"
                        height="470"
						loading="lazy"
						decoding="async"
                        src="{{ custom_theme_url('/assets/img/misc/dash-marketplace-announce.png') }}"
                    >
                </figure>
                <a
                    class="absolute inset-0 z-1 inline-block overflow-hidden text-start -indent-96"
                    href="{{ route('dashboard.admin.marketplace.index') }}"
                >
                    {{ __('Explore Marketplace') }}
                </a>
            </x-card>
            {{-- end: brand --}}

            {{-- begin: group-widgets --}}
            <div class="grid grid-cols-1 gap-8 lg:grid-cols-2 lg:gap-11">
                @php
                    $widgets = cache('dashboard_widgets', []);
                @endphp

                @foreach ($widgets as $widget)
					@if($widget->name === \App\Enums\DashboardWidget::PREMIUM_ADVANTAGES && \App\Helpers\Classes\Helper::isUserVIP())
						@continue
					@endif

                    @if ($widget->enabled)
                        @includeIf('panel.admin.dashboard.' . $widget?->name?->value, ['widget' => $widget])
                    @endif
                @endforeach
            </div>
            {{-- end: group-widgets --}}
        </div>
    </div>
@endsection

@push('script')
    <script src="{{ custom_theme_url('/assets/libs/jquery-ui/jquery-ui.min.js') }}"></script>
    <script src="{{ custom_theme_url('/assets/libs/apexcharts/dist/apexcharts.min.js') }}"></script>
    <script src="{{ custom_theme_url('/assets/libs/nested-sortable/jquery.mjs.nestedSortable.js') }}"></script>
    <script>
        $(document).ready(function() {
            const $menuList = $('.lqd-menu-list');

            $menuList.nestedSortable({
                handle: ".lqd-menu-item-handle",
                items: 'li',
                toleranceElement: '> div',
                placeholder: 'lqd-menu-item-placeholder',
                forcePlaceholderSize: true,
                maxLevels: 1,
                update: function(event, ui) {
                    let menu_serialized = $menuList.nestedSortable("serialize");
                    $.ajax({
                        type: 'PUT',
                        url: '{{ route('dashboard.admin.dashboard-widget.order') }}',
                        data: $menuList.nestedSortable("serialize"),
                        dataType: "text",
                        success: function(resultData) {
                            toastr.success(resultData.message && resultData.message.length ?
                                resultData.message : '{{ __('Updated successfully') }}'
                            );
                        }
                    });

                    const $liElements = $menuList.children('li');

                    $liElements.each(function(index) {
                        const $el = $(this);
                        const name = $el.attr('data-name');
                        const $targetCard = $(`#admin-card-${name}`);

                        if (!$targetCard.length) return;

                        $targetCard.css('order', index);
                    });
                },
            });

            $("[data-status='menu']").on('change', function() {
                const $checkbox = $(this);
                const route = $checkbox.data('href');
                const name = $checkbox.attr('data-name');
                const $targetCard = $(`#admin-card-${name}`);

                $.ajax({
                    type: 'PUT',
                    url: route,
                    data: {
                        _token: "{{ csrf_token() }}"
                    },
                    dataType: "json",
                    success: function(resultData) {
                        toastr.success(resultData.message);
                    }
                });

                if ($targetCard.length) {
                    $targetCard.css('display', $checkbox.is(':checked') ? 'block' : 'none');
                }
            });
        });
    </script>
    <script>
        (() => {
            "use strict";

            function mapRange(value, in_min, in_max, out_min, out_max) {
                return ((value - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min;
            }

            @php
                $daily_sales = json_decode(cache('daily_sales'));

                if (empty($daily_sales) || !is_array($daily_sales)) {
                    $daily_sales = [];
                }

				if (\App\Helpers\Classes\Helper::appIsDemo()) {
					$daily_sales = json_decode(json_encode(\App\Helpers\Classes\Helper::generateFakeDataLastMonth()));
				}

                $top_countries = json_decode(cache('top_countries'));
                if (empty($top_countries) || !is_array($top_countries)) {
                    $top_countries = [];
                }

				if (\App\Helpers\Classes\Helper::appIsDemo()) {
					$top_countries = json_decode(\App\Helpers\Classes\Helper::demoDataForAdminDashboardTopCountries());
				}

                $user_traffic = json_decode(cache('user_traffic'));
                if (empty($user_traffic) || !is_array($user_traffic)) {
                    $user_traffic = [];
                }

				if (\App\Helpers\Classes\Helper::appIsDemo()) {
					$user_traffic = json_decode(\App\Helpers\Classes\Helper::demoDataForAdminDashboardUserTraffic());

				}

                $new_customers = json_decode(cache('new_customers'));

				if(\App\Helpers\Classes\Helper::appIsDemo()) {
					$new_customers = json_decode(json_encode(\App\Helpers\Classes\Helper::generateFakeDataNewCustomer()));
				}
            @endphp

            // Start Sales Chart
            const currentDate = new Date();
            const targetDate = new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, currentDate.getDate());

            const dailySalesChartOptions = {
                series: [{
                    name: 'Sales',
                    data: [
                        @foreach ($daily_sales as $dailySales)
                            [{{ strtotime($dailySales->days) * 1000 }}, {{ $dailySales->sums }}],
                        @endforeach
                    ]
                }],
                chart: {
                    id: 'area-datetime',
                    type: 'area',
                    height: 210,
                    zoom: {
                        enabled: false
                    },
                    toolbar: {
                        show: false
                    }
                },
                dataLabels: {
                    enabled: false
                },
                grid: {
                    show: false,
                },
                stroke: {
                    show: false,
                },
                xaxis: {
                    type: 'datetime',
                    labels: {
                        offsetY: 0,
                        style: {
                            colors: 'hsl(var(--foreground) / 40%)',
                            fontSize: '10px',
                            fontFamily: 'inherit',
                            fontWeight: 500,
                        },
                    },
                    axisBorder: {
                        show: false,
                    },
                    axisTicks: {
                        show: false,
                    },
                    min: targetDate.getTime(),
                    max: currentDate.getTime()
                },
                yaxis: {
                    labels: {
                        offsetX: -15,
                        style: {
                            colors: 'hsl(var(--foreground) / 40%)',
                            fontSize: '10px',
                            fontFamily: 'inherit',
                            fontWeight: 500,
                        },
                    },
                    axisBorder: {
                        show: false,
                    },
                    axisTicks: {
                        show: false,
                    },
                },
                tooltip: {
                    x: {
                        format: 'dd MMM yyyy'
                    }
                },
                stroke: {
                    width: 2,
                    colors: ['#BCA8F0'],
                    curve: 'smooth'
                },
                fill: {
                    type: 'gradient',
                    gradient: {
                        shadeIntensity: 1,
                        opacityFrom: 0.3,
                        opacityTo: 0.6,
                        stops: [0, 100],
                        colorStops: [
                            [{
                                    offset: 50,
                                    color: '#EAE8FA',
                                    opacity: 0.1
                                },
                                {
                                    offset: 150,
                                    color: '#6A22C5',
                                    opacity: 0
                                },
                            ]
                        ]
                    }
                },
            };

            if (document.querySelector("#chart-daily-sales")) {
                const dailySalesChart = new ApexCharts(document.querySelector("#chart-daily-sales"),
                    dailySalesChartOptions);
                dailySalesChart.render();

                window.updateDateRange = function(e, type = 'day') {
                    const currentDate = new Date();
                    const targetDate = new Date();

                    $('.finance-change-range-btn').removeClass('active');
                    $(e.currentTarget).addClass('active');

                    if (type == 'day') {
                        targetDate.setDate(targetDate.getDate() - 1);
                    } else if (type == 'week') {
                        targetDate.setDate(targetDate.getDate() - 7);
                    } else if (type == 'month') {
                        targetDate.setMonth(targetDate.getMonth() - 1);
                    } else if (type == 'year') {
                        targetDate.setFullYear(targetDate.getFullYear() - 1);
                    }
                    dailySalesChart.updateOptions({
                        xaxis: {
                            min: targetDate.getTime(),
                            max: currentDate.getTime()
                        }
                    });
                }
            }
            // End Sales Chart

			@if(\App\Helpers\Classes\Helper::appIsDemo())
				@php
					$popular_plans_data = \App\Helpers\Classes\Helper::demoDataForAdminDashboardPopularPlans();
				@endphp
			@endif

            // Start Popular Plans Chart
            const data = @json($popular_plans_data);
            const dataLength = data.length;
            const series = [];
            const minBubbleRadius = 40;
            const maxBubbleRadius = 90;
            let biggestValue = data[0];
            let total = 0;

            // first, add invisible data in all 4 corners to prevent overflow hidden
            series.push({
                name: '',
                data: [
                    [-2, -2, 0]
                ],
                color: '#ffffff00'
            }, {
                name: '',
                data: [
                    [dataLength + 2, -2, 0]
                ],
                color: '#ffffff00'
            }, {
                name: '',
                data: [
                    [dataLength + 2, dataLength + 2, 0]
                ],
                color: '#ffffff00'
            }, {
                name: '',
                data: [
                    [-2, dataLength + 2, 0]
                ],
                color: '#ffffff00'
            });


            for (let i = 1; i < dataLength; i++) {
                total += (data[i]?.value || 0);

                if (data[i]?.value > biggestValue?.value) {
                    biggestValue = data[i];
                }
            }

            // Calculate the coordinates for the biggest value
            let centerX = dataLength <= 1 ? 0.5 : Math.round((dataLength / 2) - 0.5);
            let centerY = dataLength <= 1 ? 0.5 : Math.round(dataLength / 2);

            // Calculate the remaining coordinates
            let angle = 0;
            let angleIncrement = (2 * Math.PI) / dataLength;
            for (let i = 0; i < dataLength; i++) {
                const isBiggestValue = data[i]?.label === biggestValue?.label && data[i]?.value === biggestValue?.value;
                let radius = Math.random() + 2;
                let x = centerX + radius * Math.cos(angle);
                let y = Math.min(dataLength, centerY + radius * Math.sin(angle));

                if (isBiggestValue) {
                    x = centerX;
                    y = centerY;
                }

                series.push({
                    name: data[i]?.label,
                    data: [
                        [x, y, mapRange(data[i]?.value, 0, biggestValue?.value, minBubbleRadius,
                            maxBubbleRadius)]
                    ],
                    color: data[i]?.color
                });

                angle += angleIncrement;
            }

            const popularPlansChartOptions = {
                series,
                chart: {
                    height: 300,
                    type: 'bubble',
                    dropShadow: {
                        enabled: false,
                        enabledOnSeries: undefined,
                        top: 4,
                        left: 0,
                        blur: 6,
                        color: '#000',
                        opacity: 0.1
                    },
                    zoom: {
                        enabled: false
                    },
                    toolbar: {
                        show: false
                    }
                },
                dataLabels: {
                    enabled: true,
                    formatter: function(val, opts) {
                        if (typeof val === 'undefined' || opts.seriesIndex <= 3) {
                            return '';
                        }

                        let percentage = Math.round((data[opts.seriesIndex - 4]?.value / total) * 100);

                        if (isNaN(percentage)) {
                            percentage = 0;
                        }

                        return `${percentage}%`;
                    },
                    style: {
                        fontFamily: 'var(--font-heading)',
                        fontSize: '18px',
                        fontWeight: 600,
                        colors: ['hsl(var(--foreground))'],
                    },
                },
                plotOptions: {
                    bubble: {
                        zScaling: false,
                        minBubbleRadius,
                        maxBubbleRadius,
                    }
                },
                grid: {
                    show: false,
                },
                stroke: {
                    show: false,
                    width: 0
                },
                markers: {
                    strokeWidth: 0,
                },
                tooltip: {
                    enabled: false
                },
                xaxis: {
                    labels: {
                        show: false
                    },
                    axisBorder: {
                        show: false,
                    },
                    axisTicks: {
                        show: false,
                    }
                },
                yaxis: {
                    labels: {
                        show: false
                    },
                    axisBorder: {
                        show: false,
                    },
                    axisTicks: {
                        show: false,
                    }
                },
                legend: {
                    show: true,
                    formatter: function(seriesName) {
                        return seriesName;
                    }
                }
            };

            if (document.querySelector("#revenue-source")) {
                const popularPlansChart = new ApexCharts(document.querySelector("#revenue-source"),
                    popularPlansChartOptions);
                popularPlansChart.render();
            }
            // End Popular Plans Chart

            // Start API Cost Distribution = Popular Tools Data
            const apiCostDistribution = @json($api_cost_distribution);
            const apiCostDistributionOptions = {
                series: [{
                    name: '{{ __('Percent') }}',
                    data: []
                }],
                colors: ['hsl(var(--secondary))'],
                chart: {
                    type: 'bar',
                    height: 350,
                    zoom: {
                        enabled: false
                    },
                    toolbar: {
                        show: false
                    }
                },
                plotOptions: {
                    bar: {
                        borderRadius: 4,
                        horizontal: true,
                        barHeight: '35px'
                    }
                },
                grid: {
                    show: false
                },
                stroke: {
                    show: false,
                    width: 0
                },
                dataLabels: {
                    enabled: false
                },
                xaxis: {
                    categories: [],
                    axisBorder: {
                        show: false,
                    },
                    axisTicks: {
                        show: false,
                    }
                },
                yaxis: {
                    labels: {
                        show: true,
                        style: {
                            fontFamily: 'var(--font-heading)',
                            fontSize: '15px',
                            fontWeight: 500,
                            colors: ['hsl(var(--foreground) / 50%)'],
                        }
                    },
                    axisBorder: {
                        show: false,
                    },
                    axisTicks: {
                        show: false,
                    }
                },
                states: {
                    hover: {
                        filter: {
                            type: 'none'
                        }
                    }
                },
            };

            apiCostDistribution.forEach(element => {
                apiCostDistributionOptions.series[0].data.push(Number(element.value));
                apiCostDistributionOptions.xaxis.categories.push(element.label);
            });

            if (document.querySelector("#api-cost-distribution")) {
                const apiCostDistributionChart = new ApexCharts(document.querySelector("#api-cost-distribution"),
                    apiCostDistributionOptions);
                apiCostDistributionChart.render();
            }
            // End API Cost Distribution

            // Start Top Countries
            const topCountries = @json($top_countries);
            const topCountriesChartOptions = {
                series: [{
                    name: 'Users',
                    data: []
                }],
                colors: ['hsl(var(--secondary))'],
                chart: {
                    type: 'bar',
                    height: 350,
                    zoom: {
                        enabled: false
                    },
                    toolbar: {
                        show: false
                    }
                },
                plotOptions: {
                    bar: {
                        borderRadius: 4,
                        horizontal: true,
                        dataLabels: {
                            position: 'bottom'
                        },
                        barHeight: '35px'
                    }
                },
                grid: {
                    show: false
                },
                stroke: {
                    show: false,
                    width: 0
                },
                dataLabels: {
                    enabled: true,
                    textAnchor: 'start',
                    formatter: (val, opt) => {
                        return topCountries[opt.dataPointIndex].country ?? 'Unknown';
                    },
                    style: {
                        fontFamily: 'var(--font-heading)',
                        fontSize: '15px',
                        fontWeight: 500,
                        colors: ['hsl(var(--secondary-foreground))'],
                    },

                },
                xaxis: {
                    categories: [],
                    labels: {
                        show: false
                    },
                    axisBorder: {
                        show: false,
                    },
                    axisTicks: {
                        show: false,
                    }
                },
                yaxis: {
                    opposite: true,
                    labels: {
                        show: true,
                        formatter: (val) => {
                            return val
                        },
                        style: {
                            fontFamily: 'var(--font-heading)',
                            fontSize: '15px',
                            fontWeight: 500,
                            colors: ['hsl(var(--foreground) / 40%)'],
                        }
                    },
                    axisBorder: {
                        show: false,
                    },
                    axisTicks: {
                        show: false,
                    }
                },
                fill: {
                    opacity: 1
                },
                states: {
                    hover: {
                        filter: {
                            type: 'none'
                        }
                    }
                },
            };

            topCountries.forEach(element => {
                topCountriesChartOptions.series[0].data.push(Number(element.total));
                topCountriesChartOptions.xaxis.categories.push(Number(element.total, 0));
            });

            if (document.querySelector("#top-countries")) {
                const topCountriesChart = new ApexCharts(document.querySelector("#top-countries"),
                    topCountriesChartOptions);
                topCountriesChart.render();
            }
            // End Top countries

            // Start Popular Tools Chart
            const popularToolsData = @json($popular_tools_data);
            const popularToolsChartOptions = {
                series: [],
                labels: [],
                colors: [],
                chart: {
                    height: 210,
                    type: 'donut',
                },
                plotOptions: {
                    pie: {
                        donut: {
                            size: '90%',
                            labels: {
                                show: true,
                                name: {
                                    show: false
                                },
                                value: {
                                    show: true,
                                    fontSize: '36px',
                                    fontFamily: 'var(--headings-font-family)',
                                    fontWeight: 700,
                                    color: 'hsl(var(--heading-foreground)/70%)',
                                    formatter: function(val) {
                                        return `${val}%`;
                                    },
                                },
                            }
                        }
                    },
                },
                dataLabels: {
                    enabled: false
                },
                legend: {
                    fontSize: '14px',
                    fontFamily: 'var(--font-body)',
                    fontWeight: 400,
                    formatter: function(seriesName, opts) {
                        return [seriesName, `<span>${opts.w.globals.series[opts.seriesIndex]}%</span>`];
                    },
                    markers: {
                        width: 8,
                        height: 8,
                        radius: 2,
                    },
                    itemMargin: {
                        horizontal: 0,
                        vertical: 0
                    }
                },
                stroke: {
                    colors: ['hsl(var(--border))']
                },
                responsive: [{
                    breakpoint: 769,
                    options: {
                        legend: {
                            position: 'bottom',
                        }
                    }
                }, {
                    breakpoint: 501,
                    options: {
                        chart: {
                            height: 500,
                        },
                        legend: {
                            position: 'bottom',
                        }
                    }
                }]
            };

            popularToolsData.forEach(tool => {
                popularToolsChartOptions.series.push(Number(tool.value));
                popularToolsChartOptions.labels.push(tool.label);
                popularToolsChartOptions.colors.push(tool.color);
            });

            if (document.querySelector("#popular-ai-tools")) {
                const popularToolsChart = new ApexCharts(document.querySelector("#popular-ai-tools"),
                    popularToolsChartOptions);
                popularToolsChart.render();
            }
            // End Popular Tools Chart

            // Start User Traffic
            const userTraffic = @json($user_traffic);
            const userTrafficChartOptions = {
                series: [{
                    name: 'Visit',
                    data: []
                }],
                colors: ['hsl(var(--secondary))'],
                chart: {
                    type: 'bar',
                    height: 350,
                    zoom: {
                        enabled: false
                    },
                    toolbar: {
                        show: false
                    }
                },
                plotOptions: {
                    bar: {
                        borderRadius: 4,
                        horizontal: true,
                        dataLabels: {
                            position: 'bottom'
                        },
                        barHeight: '35px'
                    }
                },
                grid: {
                    show: false
                },
                stroke: {
                    show: false,
                    width: 0
                },
                dataLabels: {
                    enabled: false
                },
                dataLabels: {
                    enabled: true,
                    textAnchor: 'start',
                    formatter: (val, opt) => {
                        return userTraffic[opt.dataPointIndex].domain;
                    },
                    style: {
                        fontFamily: 'var(--font-heading)',
                        fontSize: '15px',
                        fontWeight: 500,
                        colors: ['hsl(var(--secondary-foreground))'],
                    },
                },
                xaxis: {
                    categories: [],
                    labels: {
                        show: false
                    },
                    axisBorder: {
                        show: false
                    },
                    axisTicks: {
                        show: false,
                    }
                },
                yaxis: {
                    opposite: true,
                    labels: {
                        show: true,
                        formatter: (val) => {
                            return val
                        },
                        style: {
                            fontFamily: 'var(--font-heading)',
                            fontSize: '15px',
                            fontWeight: 500,
                            colors: ['hsl(var(--foreground) / 40%)'],
                        }
                    },
                    axisBorder: {
                        show: false
                    },
                    axisTicks: {
                        show: false,
                    }
                },
                fill: {
                    opacity: 1
                },
                states: {
                    hover: {
                        filter: {
                            type: 'none'
                        }
                    }
                },
            };

            userTraffic.forEach(element => {
                userTrafficChartOptions.series[0].data.push(Number(element.users));
                userTrafficChartOptions.xaxis.categories.push(Number(element.users));
            });

            if (document.querySelector("#user-traffic")) {
                const userTrafficChart = new ApexCharts(document.querySelector("#user-traffic"),
                    userTrafficChartOptions);
                userTrafficChart.render();
            }
            // End User Traffic

            // Start System Status
            const availablePercentage = @json(cache('available_diskspace'));
            const systemStatusChartOptions = {
                series: [availablePercentage, 100 - availablePercentage],
                labels: [@json(__('Available')), @json(__('Used'))],
                colors: ['#20C69F', '#20C69F40'],
                tooltip: {
                    style: {
                        color: '#ffffff',
                    },
                },
                chart: {
                    type: 'donut',
                    height: 250
                },
                legend: {
                    position: 'bottom',
                    fontFamily: 'inherit',
                },
                plotOptions: {
                    pie: {
                        startAngle: -90,
                        endAngle: 90,
                        offsetY: 0,
                        donut: {
                            size: '78%',
                        }
                    },
                },
                grid: {
                    padding: {
                        bottom: -130
                    }
                },
                stroke: {
                    width: 5,
                    colors: 'hsl(var(--surface-background))'
                },
                responsive: [{
                    breakpoint: 480,
                    options: {
                        chart: {
                            width: document.getElementById('system-status')?.offsetWidth,
                        },
                    }
                }],
                dataLabels: {
                    enabled: false,
                }
            };

            if (document.querySelector("#system-status")) {
                const systemStatusChart = new ApexCharts(document.querySelector("#system-status"),
                    systemStatusChartOptions);
                systemStatusChart.render();
            }
            // End System Status

            // Start New Customer
            const newCustomers = @json($new_customers);
            const newCustomerOptions = {
                series: [{
                    name: 'Free',
                    data: []
                }, {
                    name: 'Paid',
                    data: []
                }],
                colors: ['hsl(var(--secondary))', 'hsl(var(--primary))'],
                chart: {
                    type: 'bar',
                    height: 350,
                    zoom: {
                        enabled: false
                    },
                    toolbar: {
                        show: false
                    },
                    stacked: true
                },
                grid: {
                    show: false
                },
                stroke: {
                    width: 5,
                    colors: ['hsl(var(--card-background))']
                },
                dataLabels: {
                    enabled: false
                },
                plotOptions: {
                    bar: {
                        borderRadius: 4,
                    }
                },
                xaxis: {
                    categories: [],
                    labels: {
                        show: true
                    },
                    axisBorder: {
                        show: false,
                    },
                    axisTicks: {
                        show: false,
                    }
                },
                yaxis: {
                    labels: {
                        show: true,
                        formatter: function(value) {
                            return Math.round(value);
                        }
                    },
                    axisBorder: {
                        show: false,
                    },
                    axisTicks: {
                        show: false,
                    }
                },
                fill: {
                    opacity: 1
                },
                legend: {
                    show: false
                },
                states: {
                    hover: {
                        filter: {
                            type: 'none'
                        }
                    }
                },
            };

            Object.values(newCustomers).forEach(element => {
                newCustomerOptions.series[0].data.push(Number(element.free));
                newCustomerOptions.series[1].data.push(Number(element.paid));
                newCustomerOptions.xaxis.categories.push(element.date?.split('-')[2]);
            });

            if (document.querySelector("#new-customers")) {
                const newCustomerChart = new ApexCharts(document.querySelector("#new-customers"),
                    newCustomerOptions);
                newCustomerChart.render();
            }
            // End New Customer
        })();
    </script>
@endpush
