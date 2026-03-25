@extends('layouts.app')

@section('title', __('accounting::lang.account_payable_ageing_report'))

@section('content')

@include('accounting::layouts.nav')
<br><br>
<!-- Content Header (Page header) -->
<section class="content">
    <div class="row">
        <div class="col-md-3 col-md-offset-1">
            <div class="form-group">
                {!! Form::label('location_id',  __('purchase.business_location') . ':') !!}
                {!! Form::select('location_id', $business_locations, request()->input('location_id'), 
                    ['class' => 'form-control select2', 'style' => 'width:100%']); !!}
            </div>
        </div>
    </div>
    <div class="row">
        <div class="col-md-10 col-md-offset-1">
            <div class="box box-warning">
                <div class="box-header with-border text-center">
                    <h2 class="box-title">@lang( 'accounting::lang.account_payable_ageing_report' )</h2>
                </div>
                <div class="box-body">
                    <table class="table table-stripped table-bordered">
                        <thead>
                            <tr>
                                <th>@lang( 'lang_v1.supplier_name')</th>
                                <th style="color: #2dce89 !important;">@lang( 'lang_v1.current')</th>
                                <th style="color: #ffd026 !important;">
                                    @lang( 'accounting::lang.1_30_days' )
                                </th>
                                <th style="color: #ffa100 !important;">
                                    @lang( 'accounting::lang.31_60_days' )
                                </th>
                                <th style="color: #f5365c !important;">
                                    @lang( 'accounting::lang.61_90_days' )
                                </th>
                                <th style="color: #FF0000 !important;">
                                    @lang( 'accounting::lang.91_and_over' )
                                </th>
                                <th>@lang('sale.total')</th>
                            </tr>
                        </thead>
                        <tbody>
                            @php
                                $total_current = 0;
                                $total_1_30 = 0;
                                $total_31_60 = 0;
                                $total_61_90 = 0;
                                $total_greater_than_90 = 0;
                                $grand_total = 0;
                            @endphp
                            @foreach($report_details as $report)
                                <tr>
                                    @php
                                        $total_current += $report['<1'];
                                        $total_1_30 += $report['1_30'];
                                        $total_31_60 += $report['31_60'];
                                        $total_61_90 += $report['61_90'];
                                        $total_greater_than_90 += $report['>90'];
                                        $grand_total += $report['total_due'];
                                    @endphp
                                    <td>
                                        {{$report['name']}}
                                    </td>
                                    <td>
                                        @format_currency($report['<1'])
                                    </td>
                                    <td>
                                        @format_currency($report['1_30'])
                                    </td>
                                    <td>
                                        @format_currency($report['31_60'])
                                    </td>
                                    <td>
                                        @format_currency($report['61_90'])
                                    </td>
                                    <td>
                                        @format_currency($report['>90'])
                                    </td>
                                    <td>
                                        @format_currency($report['total_due'])
                                    </td>
                                </tr>
                            @endforeach
                        </tbody>
                        <tfoot>
                            <tr>
                                <th>
                                    @lang('sale.total')
                                </th>
                                <td>
                                    @format_currency($total_current)
                                </td>
                                <td>
                                    @format_currency($total_1_30)
                                </td>
                                <td>
                                    @format_currency($total_31_60)
                                </td>
                                <td>
                                    @format_currency($total_61_90)
                                </td>
                                <td>
                                    @format_currency($total_greater_than_90)
                                </td>
                                <td>@format_currency($grand_total)</td>
                            </tr>
                        </tfoot>
                    </table>
                </div>
            </div>
        </div>
    </div>

</section>

@stop

@section('javascript')

<script type="text/javascript">
    $(document).ready(function(){
        $('#location_id').change( function(){
            if($(this).val()) {
                window.location.href = "{{route('accounting.account_payable_ageing_report')}}?location_id=" + $(this).val();
            } else {
                window.location.href = "{{route('accounting.account_payable_ageing_report')}}";
            }
            
        });
    });
</script>

@stop