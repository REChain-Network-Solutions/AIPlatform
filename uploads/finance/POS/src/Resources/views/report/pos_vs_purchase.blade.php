@extends('layouts.main')
@section('page-title')
    {{__('POS Vs Purchase')}}
@endsection

@section('page-breadcrumb')
    {{ __('Report') }},
    {{ __('POS Vs Purchase') }}
@endsection

@push('scripts')
    <script src="{{ asset('assets/js/plugins/apexcharts.min.js') }}"></script>

    <script src="{{ asset('packages/workdo/Pos/src/Resources/assets/js/html2pdf.bundle.min.js') }}"></script>

    <script>
        (function () {
            var chartBarOptions = {
                series: [
                    {
                        name: '{{ __("Profit") }}',
                        data:  {!! json_encode($profits) !!},
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
                colors: ['#ffa21d', '#FF3A6E'],

                grid: {
                    strokeDashArray: 4,
                },
                legend: {
                    show: false,
                },
                yaxis: {
                    title: {
                        text: '{{ __("Profit") }}'
                    },

                }

            };
            var arChart = new ApexCharts(document.querySelector("#pos-vs-purchase"), chartBarOptions);
            arChart.render();
        })();
    </script>
    <script type="text/javascript" src="{{ asset('js/html2pdf.bundle.min.js') }}"></script>
    <script>
        var year = '{{$currentYear}}';
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
@endpush


@section('page-action')
    <div class="float-end">
        <a href="#" class="btn btn-sm btn-primary" onclick="saveAsPDF()"data-bs-toggle="tooltip" title="{{__('Download')}}"
           data-original-title="{{__('Download')}}">
            <span class="btn-inner--icon"><i class="ti ti-download"></i></span>
        </a>
    </div>
@endsection

@section('content')
    <div class="row">
        <div class="col-sm-12">
            <div class="mt-2 " id="multiCollapseExample1">
                <div class="card">
                    <div class="card-body">
                        {{ Form::open(array('route' => array('report.pos.vs.purchase'),'method' => 'GET','id'=>'pos_vs_purchase')) }}
                        <div class="row align-items-center justify-content-end">
                            <div class="col-xl-10">
                                <div class="row ">
                                    <div class="col-xl-3 col-lg-3 col-md-6 col-sm-12 col-12">
                                        <div class="btn-box">
                                        </div>
                                    </div>

                                    <div class="col-xl-3 col-lg-3 col-md-6 col-sm-12 col-12">
                                        <div class="btn-box">
                                        </div>
                                    </div>
                                    <div class="col-xl-3 col-lg-3 col-md-6 col-sm-12 col-12">
                                        <div class="btn-box">
                                        </div>
                                    </div>
                                    <div class="col-xl-3 col-lg-3 col-md-6 col-sm-12 col-12">
                                        <div class="btn-box">
                                            {{ Form::label('year', __('Year'),['class'=>'form-label'])}}
                                            {{ Form::select('year',$yearList,isset($_GET['year'])?$_GET['year']:'', array('class' => 'form-control select')) }}
                                        </div>
                                    </div>

                                </div>
                            </div>
                            <div class="col-auto">
                                <div class="row">
                                    <div class="col-auto mt-4 d-flex">
                                        <a href="#" class="btn btn-sm btn-primary me-2" onclick="document.getElementById('pos_vs_purchase').submit(); return false;" data-bs-toggle="tooltip" title="{{__('Apply')}}" data-original-title="{{__('apply')}}">
                                            <span class="btn-inner--icon"><i class="ti ti-search"></i></span>
                                        </a>
                                        <a href="{{route('report.pos.vs.purchase')}}" class="btn btn-sm btn-danger " data-bs-toggle="tooltip"  title="{{ __('Reset') }}" data-original-title="{{__('Reset')}}">
                                            <span class="btn-inner--icon"><i class="ti ti-trash-off text-white-off "></i></span>
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    {{ Form::close() }}
                </div>
            </div>
        </div>
    </div>


    <div id="printableArea">
        <div class="row mt-0">
            <div class="col">
                <input type="hidden" value="{{__('POS Vs Purchase').' '.'Report of'.' '.$filter['startDateRange'].' to '.$filter['endDateRange']}}" id="filename">
                <div class="card p-4 mb-4">
                    <h7 class="report-text gray-text mb-0">{{__('Report')}} :</h7>
                    <h6 class="report-text mb-0">{{__('POS Vs Purchase')}}</h6>
                </div>
            </div>
            <div class="col">
                <div class="card p-4 mb-4">
                    <h7 class="report-text gray-text mb-0">{{__('Duration')}} :</h7>
                    <h6 class="report-text mb-0">{{$filter['startDateRange'].' to '.$filter['endDateRange']}}</h6>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-12" id="chart-container">
                <div class="card">
                    <div class="card-body">
                        <div class="scrollbar-inner">
                            <div id="pos-vs-purchase" data-color="primary" data-height="300"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="col-12">
            <div class="card">
                <div class="card-body table-border-style">
                    <div class="table-responsive">
                        <table class="table">
                            <thead>
                            <tr>
                                <th>{{__('Type')}}</th>
                                @foreach($monthList as $month)
                                    <th>{{$month}}</th>
                                @endforeach
                            </tr>
                            </thead>
                            <tbody>
                            <tr>
                                <td>{{(__('POS'))}}</td>
                                @foreach($posTotal as $pos)
                                    <td>{{currency_format_with_sym($pos)}}</td>
                                @endforeach
                            </tr>
                            <tr>
                                <td>{{(__('Purchase'))}}</td>
                                @foreach($purchaseTotal as $purchase)
                                    <td>{{currency_format_with_sym($purchase)}}</td>
                                @endforeach
                            </tr>
                            <tr>
                                <td colspan="13" class="text-dark"><span>{{__('Profit = POS - Purchase')}}</span></td>
                            </tr>
                            <tr>
                                <td><h6>{{(__('Profit'))}}</h6></td>

                                @foreach($profits as $profit)
                                    <td>{{currency_format_with_sym(str_replace(",", "", $profit))}}</td>
                                @endforeach
                            </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

    </div>

@endsection


