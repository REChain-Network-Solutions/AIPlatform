@extends('layouts.main')

@section('page-title')
    {{__('Dashboard')}}
@endsection

@section('page-breadcrumb')
    {{ __('POS')}}
@endsection


@section('content')
    <div class="row">
        @if (count($lowstockproducts) > 0)
            <div class="col-md-12">
                @foreach ($lowstockproducts as $product)
                    <div class="alert alert-warning alert-dismissible fade show" role="alert">
                        <span class="alert-icon"><i class="ti ti-alert-triangle"></i></span>
                        <strong>{{ $product['name'] }}</strong><small>{{ __(' (Only ') . $product['quantity'] . __(' items left)') }}</small>
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                @endforeach
            </div>
        @endif
    </div>

    @if ( $productscount == 0 || $customers == 0 || $vendors == 0)
        <div class="row">
            <div class="col-md-12">
                <?php
                $alerts = [];
                $alerts[] = $productscount == 0 ? __('Please add some Products!') : '';

                $alerts[] = $customers == 0 ? __('Please add some Customers!') : '';

                $alerts[] = $vendors == 0 ? __('Please add some Vendors!') : '';

                $result = array_filter($alerts);
                ?>
                @if (isset($result) && !empty($result) && count($result) > 0)
                    @foreach ($result as $alert)
                        <div class="alert alert-warning alert-dismissible fade show  mt-1" role="alert">
                            <span class="alert-icon"><i class="ti ti-alert-triangle"></i></span>
                            <strong>{{ $alert }}</strong>
                            <button type="button" class="btn-close" data-bs-dismiss="alert"
                                aria-label="Close"></button>
                        </div>
                    @endforeach
                @endif
            </div>
        </div>
    @endif

    <div class="row row-gap mb-4">
        <div class="col-xxl-6 col-12">
            <div class="dashboard-card">
                <img src="{{ asset('assets/images/layer.png')}}" class="dashboard-card-layer" alt="layer">
                <div class="card-inner">
                    <div class="card-content">
                        <h2> {{ $ActiveWorkspaceName}} </h2>
                        <p>{{__('Streamline sales and inventory with our all-in-one POS solution for enhanced business efficiency')}}</p>
                    </div>
                    <div class="card-icon  d-flex align-items-center justify-content-center">
                        <svg xmlns="http://www.w3.org/2000/svg" width="78" height="77" viewBox="0 0 78 77" fill="none">
                        <path opacity="0.6" d="M74.3271 26.1686C72.7024 26.1686 71.3838 24.8548 71.3838 23.236V9.55067C71.3838 7.24762 70.752 6.61809 68.4405 6.61809H54.7052C53.0805 6.61809 51.762 5.30429 51.762 3.68551C51.762 2.06673 53.0805 0.75293 54.7052 0.75293H68.4405C73.97 0.75293 77.2704 4.04133 77.2704 9.55067V23.236C77.2704 24.8548 75.9518 26.1686 74.3271 26.1686ZM6.63168 23.236V9.55067C6.63168 7.24762 7.2635 6.61809 9.57495 6.61809H23.3103C24.9349 6.61809 26.2535 5.30429 26.2535 3.68551C26.2535 2.06673 24.9349 0.75293 23.3103 0.75293H9.57495C4.04551 0.75293 0.745117 4.04133 0.745117 9.55067V23.236C0.745117 24.8548 2.06371 26.1686 3.6884 26.1686C5.31309 26.1686 6.63168 24.8548 6.63168 23.236ZM26.2535 74.0674C26.2535 72.4486 24.9349 71.1348 23.3103 71.1348H9.57495C7.2635 71.1348 6.63168 70.5053 6.63168 68.2023V54.5169C6.63168 52.8981 5.31309 51.5843 3.6884 51.5843C2.06371 51.5843 0.745117 52.8981 0.745117 54.5169V68.2023C0.745117 73.7116 4.04551 77 9.57495 77H23.3103C24.9349 77 26.2535 75.6862 26.2535 74.0674ZM77.2704 68.2023V54.5169C77.2704 52.8981 75.9518 51.5843 74.3271 51.5843C72.7024 51.5843 71.3838 52.8981 71.3838 54.5169V68.2023C71.3838 70.5053 70.752 71.1348 68.4405 71.1348H54.7052C53.0805 71.1348 51.762 72.4486 51.762 74.0674C51.762 75.6862 53.0805 77 54.7052 77H68.4405C73.97 77 77.2704 73.7116 77.2704 68.2023Z" fill="#18BF6B"/>
                        <path opacity="0.6" d="M60.5929 38.876V51.0755C60.5929 57.058 57.2964 60.3816 51.2529 60.3816H26.7648C20.7213 60.3816 17.4248 57.058 17.4248 51.0755V38.876H60.5929Z" fill="#18BF6B"/>
                        <path d="M66.4792 41.8085H11.538C9.91332 41.8085 8.59473 40.4947 8.59473 38.8759C8.59473 37.2572 9.91332 35.9434 11.538 35.9434H66.4792C68.1039 35.9434 69.4225 37.2572 69.4225 38.8759C69.4225 40.4947 68.1039 41.8085 66.4792 41.8085Z" fill="#18BF6B"/>
                        <path d="M58.6284 30.078C57.0037 30.078 55.6851 28.7642 55.6851 27.1454V26.6762C55.6851 23.3721 54.5667 22.2577 51.2506 22.2577H26.7625C23.4464 22.2577 22.328 23.3721 22.328 26.6762V27.1454C22.328 28.7642 21.0094 30.078 19.3847 30.078C17.76 30.078 16.4414 28.7642 16.4414 27.1454V26.6762C16.4414 20.1424 20.2049 16.3926 26.7625 16.3926H51.2506C57.8082 16.3926 61.5717 20.1385 61.5717 26.6762V27.1454C61.5717 28.7642 60.2531 30.078 58.6284 30.078Z" fill="#18BF6B"/>
                        </svg>
                    </div>
                </div>
            </div>
        </div>
        <div class="col-xxl-6 col-12">
            <div class="row d-flex dashboard-wrp">
                <div class="col-md-6 col-sm-6 col-12 d-flex flex-wrap">
                    <div class="dashboard-project-card">
                        <div class="card-inner  d-flex justify-content-between">
                            <div class="card-content">
                                <div class="theme-avtar bg-white">
                                    <i class="ti ti-hand-finger text-danger"></i>
                                </div>
                                <a href="{{ route('projects.index') }}"><h3 class="mt-3 mb-0 text-danger">{{ __('Sales Of This Month') }}</h3></a>
                            </div>
                            <h3 class="mb-0">{{ $monthlySelledAmount }}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 col-sm-6 col-12 d-flex flex-wrap">
                    <div class="dashboard-project-card">
                        <div class="card-inner  d-flex justify-content-between">
                            <div class="card-content">
                                <div class="theme-avtar bg-white">
                                    <i class="ti ti-chart-pie"></i>
                                </div>
                            <h3 class="mt-3 mb-0">{{ __('Total Sales Amount') }}</h3>
                            </div>
                            <h3 class="mb-0">{{ $totalSelledAmount }}</h3>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </div>
    <div class="row">
        <div class="col-sm-12">
            <div class="row">
                <div class="col-xxl-12">
                    <div class="card">
                        <div class="card-header">

                            <div class="row ">
                                <div class="col-6">
                                    <h5>{{ __('Sale Report') }}</h5>
                                </div>
                                <div class="col-6 text-end">
                                    <h6>{{ __('Last 10 Days') }}</h6>
                                </div>
                            </div>
                        </div>
                        <div class="card-body">
                            <div id="traffic-chart"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
@endsection

@push('scripts')
    <script src="{{ asset('packages/workdo/Pos/src/Resources/assets/js/apexcharts.min.js') }}"></script>
    <script src="{{ asset('packages/workdo/Pos/src/Resources/assets/js/moment.min.js') }}"></script>
    <script>
        (function() {
            var options = {
                chart: {
                    height: 350,
                    type: 'area',
                    toolbar: {
                        show: false,
                    },
                },
                dataLabels: {
                    enabled: false
                },
                stroke: {
                    width: 2,
                    curve: 'smooth'
                },
                series: [
                    {
                        name: '{{ __('Sales') }}',
                        data: {!! json_encode($salesArray['value']) !!}

                    },
                ],
                xaxis: {
                    categories: {!! json_encode($salesArray['label']) !!},
                    title: {
                        text: '{{ __('Days') }}'
                    }
                },
                colors: ['#FF3A6E', '#6fd943'],

                grid: {
                    strokeDashArray: 4,
                },
                legend: {
                    show: false,
                },
                yaxis: {
                    title: {
                        text: '{{ __('Amount') }}'
                    },
                }
            };
            var chart = new ApexCharts(document.querySelector("#traffic-chart"), options);
            chart.render();
        })();


        $(document).on('click', '.custom-checkbox .custom-control-input', function(e) {
            $.ajax({
                url: $(this).data('url'),
                method: 'PATCH',
                success: function(response) {},
                error: function(data) {
                    data = data.responseJSON;
                    show_toastr('{{ __('Error') }}', data.error, 'error')
                }
            });
        });
    </script>
@endpush



