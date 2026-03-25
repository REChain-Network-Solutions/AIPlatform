@extends('layouts.app')

@section('title', __('accounting::lang.account_payable_ageing_details'))

@section('css')
<style>
.table-sticky thead,
.table-sticky tfoot {
  position: sticky;
}
.table-sticky thead {
  inset-block-start: 0; /* "top" */
}
.table-sticky tfoot {
  inset-block-end: 0; /* "bottom" */
}
.collapsed .collapse-tr {
    display: none;
}
</style>
@endsection

@section('content')

@include('accounting::layouts.nav')

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
                    <h2 class="box-title">@lang( 'accounting::lang.account_payable_ageing_details' )</h2>
                </div>
                <div class="box-body">
                    <table class="table table-stripped table-bordered">
                        <thead>
                            <tr>
                                <th>@lang('messages.date')</th>
                                <th>@lang('account.transaction_type')</th>
                                <th>@lang('purchase.ref_no')</th>
                                <th>@lang('report.supplier')</th>
                                <th>@lang('lang_v1.due_date')</th>
                                <th>@lang('lang_v1.due')</th>
                            </tr>
                        </thead>
                        @foreach($report_details as $key => $value)
                        <tbody @if($loop->index != 0) class="collapsed" @endif>
                            <tr class="toggle-tr" style="cursor: pointer;">
                                <th colspan="6">
                                    <span class="collapse-icon">
                                        <i class="fas fa-arrow-circle-right"></i>
                                    </span>
                                    @if($key == 'current')
                                       <spna style="color: #2dce89 !important;"> 
                                       @lang( 'accounting::lang.current' ) </spna>
                                    @elseif($key == '1_30')
                                        <span style="color: #ffd026 !important;"> 
                                        @lang( 'accounting::lang.days_past_due', ['days' => '1 - 30'] )
                                        </span>
                                    @elseif($key == '31_60')
                                        <span style="color: #ffa100 !important;"> 
                                        @lang( 'accounting::lang.days_past_due', ['days' => '31 - 60'] )
                                        </span>
                                    @elseif($key == '61_90')
                                        <span style="color: #f5365c !important;"> 
                                            @lang( 'accounting::lang.days_past_due', ['days' => '61 - 90'] )
                                        </span>
                                    @elseif($key == '>90')
                                        <span style="color: #FF0000 !important;"> 
                                        @lang( 'accounting::lang.91_and_over_past_due' )
                                        </span>
                                    @endif
                                </th>
                            </tr>
                            @php
                                $total=0;
                            @endphp
                            @foreach($value as $details)
                                @php
                                    $total += $details['due'];
                                @endphp
                                <tr class="collapse-tr">
                                    <td>
                                        {{$details['transaction_date']}}
                                    </td>
                                    <td>
                                        @lang( 'lang_v1.purchase' )
                                    </td>
                                    <td>
                                        {{$details['ref_no']}}
                                    </td>
                                    <td>
                                        {{$details['contact_name']}}
                                    </td>
                                    <td>
                                        {{$details['due_date']}}
                                    </td>
                                    <td>
                                        @format_currency($details['due'])
                                    </td>
                                </tr>
                            @endforeach
                            <tr class="collapse-tr bg-gray">
                                <th>
                                    @if($key == 'current')
                                        @lang( 'accounting::lang.total_for_current' )
                                    @elseif($key == '1_30')
                                        @lang( 'accounting::lang.total_for_days_past_due', ['days' => '1 - 30'] )
                                    @elseif($key == '31_60')
                                        @lang( 'accounting::lang.total_for_days_past_due', ['days' => '31 - 60'] )
                                    @elseif($key == '61_90')
                                        @lang( 'accounting::lang.total_for_days_past_due', ['days' => '61 - 90'] )
                                    @elseif($key == '>90')
                                        @lang( 'accounting::lang.total_for_91_and_over' )
                                    @endif
                                </th>
                                <th></th>
                                <th></th>
                                <th></th>
                                <th></th>
                                <th>@format_currency($total)</th>
                            </tr>
                        </tbody>
                        @endforeach
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
                window.location.href = "{{route('accounting.account_payable_ageing_details')}}?location_id=" + $(this).val();
            } else {
                window.location.href = "{{route('accounting.account_payable_ageing_details')}}";
            }
        });
    });
    $(document).on('click', '.toggle-tr', function(){
        $(this).closest('tbody').toggleClass('collapsed');
        var html = $(this).closest('tbody').hasClass('collapsed') ? 
        '<i class="fas fa-arrow-circle-right"></i>' : '<i class="fas fa-arrow-circle-down"></i>';
        $(this).find('.collapse-icon').html(html);
    })
</script>

@stop