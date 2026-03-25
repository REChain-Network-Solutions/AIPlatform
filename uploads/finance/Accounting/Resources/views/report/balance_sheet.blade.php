@extends('layouts.app')

@section('title', __('accounting::lang.balance_sheet'))

@section('content')

@include('accounting::layouts.nav')

<!-- Content Header (Page header) -->
<section class="content-header">
    <h1 class="tw-text-xl md:tw-text-3xl tw-font-bold tw-text-black">@lang( 'accounting::lang.balance_sheet' )</h1>
</section>

<section class="content">

    <div class="col-md-3">
        <div class="form-group">
            {!! Form::label('date_range_filter', __('report.date_range') . ':') !!}
            {!! Form::text('date_range_filter', null, 
                ['placeholder' => __('lang_v1.select_a_date_range'), 
                'class' => 'form-control', 'readonly', 'id' => 'date_range_filter']); !!}
        </div>
    </div>

    <div class="col-md-10 col-md-offset-1">
        <div class="box box-warning">
            <div class="box-header with-border text-center">
                <h2 class="box-title">@lang( 'accounting::lang.balance_sheet')</h2>
                <p>{{@format_date($start_date)}} ~ {{@format_date($end_date)}}</p>
            </div>

            <div class="box-body">
                
                @php
                    $total_assets = 0;
                    $total_liab_owners = 0;
                @endphp

                    <table class="table table-stripped table-bordered" style="min-height: 300px">
                        <thead>
                            <tr>
                                <th class="success">@lang( 'accounting::lang.assets')</th>
                                <th class="warning">@lang( 'accounting::lang.liab_owners_capital')</th>
                            </tr>
                        </thead>

                        <tr>
                            <td class="col-md-6">
                                <table class="table">
                                    @foreach($assets as $asset)
                                        @php
                                            $total_assets += $asset->balance
                                        @endphp

                                        <tr>
                                            <th>{{$asset->name}}</th>
                                            <td>@format_currency($asset->balance)</td>
                                        </tr>
                                    @endforeach
                                </table>
                            </td>

                            <td class="col-md-6">
                                <table class="table">
                                    @foreach($liabilities as $liability)

                                        @php
                                            $total_liab_owners += $liability->balance
                                        @endphp

                                        <tr>
                                            <th>{{$liability->name}}</th>
                                            <td>@format_currency($liability->balance)</td>
                                        </tr>
                                    @endforeach

                                    @foreach($equities as $equity)
                                        @php
                                            $total_liab_owners += $equity->balance
                                        @endphp
                                        
                                        <tr>
                                            <th>{{$equity->name}}</th>
                                            <td>@format_currency($equity->balance)</td>
                                        </tr>
                                    @endforeach
                                </table>
                            </td>
                        </tr>

                        <tr>
                            <td class="col-md-6">
                                <span>
                                    <strong>@lang( 'accounting::lang.total_assets'): </strong>
                                </span>

                                <span>@format_currency($total_assets)</span>
                            </td>

                            <td class="col-md-6">
                                <span>
                                    <strong>@lang( 'accounting::lang.total_liab_owners'): </strong>
                                </span>

                                <span>@format_currency($total_liab_owners)</span>
                            </td>
                        </tr>

                    </table>
                
            </div>

        </div>
    </div>

</section>

@stop

@section('javascript')

<script type="text/javascript">
    $(document).ready(function(){

        dateRangeSettings.startDate = moment('{{$start_date}}');
        dateRangeSettings.endDate = moment('{{$end_date}}');

        $('#date_range_filter').daterangepicker(
            dateRangeSettings,
            function (start, end) {
                $('#date_range_filter').val(start.format(moment_date_format) + ' ~ ' + end.format(moment_date_format));
                apply_filter();
            }
        );
        $('#date_range_filter').on('cancel.daterangepicker', function(ev, picker) {
            $('#date_range_filter').val('');
            apply_filter();
        });

        function apply_filter(){
            var start = '';
            var end = '';

            if ($('#date_range_filter').val()) {
                start = $('input#date_range_filter')
                    .data('daterangepicker')
                    .startDate.format('YYYY-MM-DD');
                end = $('input#date_range_filter')
                    .data('daterangepicker')
                    .endDate.format('YYYY-MM-DD');
            }

            const urlParams = new URLSearchParams(window.location.search);
            urlParams.set('start_date', start);
            urlParams.set('end_date', end);
            window.location.search = urlParams;
        }
    });

</script>

@stop