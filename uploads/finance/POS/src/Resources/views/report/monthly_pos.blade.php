@extends('layouts.main')
@section('page-title')
    {{__('Manage Pos')}}
@endsection

@section('page-breadcrumb')
    {{ __('Report') }},
    {{ __('Daily Pos Report') }}
@endsection

@push('scripts')
    <script src="{{ asset('assets/js/plugins/apexcharts.min.js') }}"></script>
    <script src="{{ asset('packages/workdo/Pos/src/Resources/assets/js/html2pdf.bundle.min.js') }}"></script>
    <script>
        var filename = $('#filename').val();

        function saveAsPDF() {
            var element = document.getElementById('printableArea');
            var opt = {
                margin: 0.3,
                filename: filename,
                image: {type: 'jpeg', quality: 1},
                html2canvas: {scale: 4, dpi: 72, letterRendering: true},
                jsPDF: {unit: 'in', format: 'A2'}
            };
            html2pdf().set(opt).from(element).save();
        }
    </script>

    <script>
        (function () {
            var chartBarOptions = {
                series: [
                    {
                        name: '{{ __("Pos") }}',
                        data:   {!! json_encode($data) !!},
                    },
                ],

                chart: {
                    height: 300,
                    type: 'area',
                    dropShadow: {
                        enabled: true,
                        color: '#000',
                        top: 18,
                        left: 7,
                        blur: 10,
                        opacity: 0.2
                    },
                    toolbar: {
                        show: false
                    }
                },
                dataLabels: {
                    enabled: false
                },
                stroke: {
                    width: 2,
                    curve: 'smooth'
                },
                title: {
                    text: '',
                    align: 'left'
                },
                xaxis: {
                    categories: {!! json_encode($monthList) !!},
                    title: {
                        text: '{{ __("Months") }}'
                    }
                },
                colors: ['#6fd944', '#6fd944'],

                grid: {
                    strokeDashArray: 4,
                },
                legend: {
                    show: false,
                },
                yaxis: {
                    title: {
                        text: '{{ __("Amount") }}'
                    },

                }

            };
            var arChart = new ApexCharts(document.querySelector("#monthly-pos"), chartBarOptions);
            arChart.render();
        })();
    </script>
@endpush

@section('page-action')
    <div class="float-end">
        <a href="#" class="btn btn-sm btn-primary" onclick="saveAsPDF()"data-bs-toggle="tooltip" title="{{__('Download')}}" data-original-title="{{__('Download')}}">
            <span class="btn-inner--icon"><i class="ti ti-download"></i></span>
        </a>
    </div>
@endsection  
@section('content')

    <ul class="nav nav-pills my-3" id="pills-tab" role="tablist">
        <li class="nav-item">
            <a class="nav-link" id="pills-home-tab" data-bs-toggle="pill"
               href="{{ route('report.daily.pos') }}"
               onclick="window.location.href = '{{ route('report.daily.pos') }}'" role="tab"
               aria-controls="pills-home" aria-selected="true">{{ __('Daily') }}</a>
        </li>
        <li class="nav-item">
            <a class="nav-link active" id="pills-profile-tab" data-bs-toggle="pill" href="#monthly-chart" role="tab"
               aria-controls="pills-profile" aria-selected="false">{{ __('Monthly') }}</a>
        </li>
    </ul>

    <div class="row">
        <div class="col-sm-12">
            <div class="mt-2 " >
                <div class="card">
                    <div class="card-body">
                        {{ Form::open(['route' => ['report.monthly.pos'], 'method' => 'GET', 'id' => 'monthly_pos_report_submit']) }}
                        <div class="row d-flex align-items-center justify-content-end">
                            <div class="col-xl-3 col-lg-3 col-md-6 col-sm-12 col-12">
                                <div class="btn-box">
                                    {{ Form::label('year', __('Year'),['class'=>'form-label'])}}
                                    {{ Form::select('year',$yearList,isset($_GET['year'])?$_GET['year']:'', array('class' => 'form-control select')) }}
                                </div>
                            </div>

                            <div class="col-xl-3 col-lg-3 col-md-6 col-sm-12 col-12 mr-2">
                                <div class="btn-box">
                                    {{ Form::label('warehouse', __('Warehouse'),['class'=>'form-label'])}}
                                    {{ Form::select('warehouse',$warehouse,isset($_GET['warehouse'])?$_GET['warehouse']:'', array('class' => 'form-control select')) }}
                                </div>
                            </div>
                            <div class="col-xl-3 col-lg-3 col-md-6 col-sm-12 col-12 mr-2">
                                <div class="btn-box">
                                    {{ Form::label('customer', __('Customer'),['class'=>'form-label'])}}
                                    {{ Form::select('customer',$customer,isset($_GET['customer'])?$_GET['customer']:'', array('class' => 'form-control select')) }}
                                </div>
                            </div>

                            <div class="col-auto float-end ms-2 mt-4 d-flex">
                                <a href="#" class="btn btn-sm btn-primary me-2"
                                   onclick="document.getElementById('monthly_pos_report_submit').submit(); return false;"
                                   data-toggle="tooltip" data-original-title="{{ __('apply') }}" title="{{ __('Apply') }}">
                                    <span class="btn-inner--icon"><i class="ti ti-search"></i></span>
                                </a>
                                <a href="{{ route('report.monthly.pos') }}" class="btn btn-sm btn-danger" data-toggle="tooltip"
                                   data-original-title="{{ __('Reset') }}"  title="{{ __('Reset') }}">
                                    <span class="btn-inner--icon"><i class="ti ti-trash-off text-white-off"></i></span>
                                </a>
                            </div>

                        </div>
                        {{ Form::close() }}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div id="printableArea">
        <div class="row mt-0">
            <div class="col">
                <input type="hidden" value="{{$filter['warehouse'].' '.__('Monthly Pos').' '.'Report of'.' '.$filter['startMonth'].' to '.$filter['endMonth']}}" id="filename">
                <div class="card p-4 mb-4">
                    <h7 class="report-text gray-text mb-0">{{__('Report')}} :</h7>
                    <h6 class="report-text mb-0">{{__('Monthly Pos Report')}}</h6>
                </div>
            </div>
            @if(!empty($filter['warehouse']))

                <div class="col">
                    <div class="card p-4 mb-4">
                        <h7 class="report-text gray-text mb-0">{{__('Warehouse')}} :</h7>
                        <h6 class="report-text mb-0">{{$filter['warehouse']}}</h6>
                    </div>
                </div>
            @endif
            @if(!empty($filter['customer']))
                <div class="col">
                    <div class="card p-4 mb-4">
                        <h7 class="report-text gray-text mb-0">{{__('Customer')}} :</h7>
                        <h6 class="report-text mb-0">{{$filter['customer']}}</h6>
                    </div>
                </div>
            @endif
            <div class="col">
                <div class="card p-4 mb-4">
                    <h7 class="report-text gray-text mb-0">{{__('Duration')}} :</h7>
                    <h6 class="report-text mb-0">{{$filter['startMonth'].' to '.$filter['endMonth']}}</h6>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="setting-tab">
                        <div class="tab-content">
                            <div class="tab-pane fade show active" id="monthly-chart" role="tabpanel">
                                <div class="col-lg-12">
                                    <div class="card-header">
                                        <div class="row ">
                                            <div class="col-6">
                                                <h6>{{ __('Monthly Report') }}</h6>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="card-body">
                                        <div id="monthly-pos"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    </div>

@endsection



