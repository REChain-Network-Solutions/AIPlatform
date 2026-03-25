@extends('layouts.app')

@section('title', __('accounting::lang.ledger'))

@section('content')

@include('accounting::layouts.nav')

<!-- Content Header (Page header) -->
<section class="content-header">
    <h1 class="tw-text-xl md:tw-text-3xl tw-font-bold tw-text-black">@lang( 'accounting::lang.ledger' ) - {{$account->name}}</h1>
</section>

<section class="content">
    <div class="row">
        <div class="col-md-5">
            <div class="box box-solid">
                <div class="box-body">
                    <table class="table table-condensed">
                        <tr>
                            <th>@lang( 'user.name' ):</th>
                            <td>
                                {{$account->name}}

                                @if(!empty($account->gl_code))
                                    ({{$account->gl_code}})
                                @endif
                            </td>
                        </tr>

                        <tr>
                            <th>@lang( 'accounting::lang.account_type' ):</th>
                            <td>
                                @if(!empty($account->account_primary_type))
                                    {{__('accounting::lang.' . $account->account_primary_type)}}
                                @endif
                            </td>
                        </tr>

                        <tr>
                            <th>@lang( 'accounting::lang.account_sub_type' ):</th>
                            <td>
                                @if(!empty($account->account_sub_type))
                                    {{__('accounting::lang.' . $account->account_sub_type->name)}}
                                @endif
                            </td>
                        </tr>

                        <tr>
                            <th>@lang( 'accounting::lang.detail_type' ):</th>
                            <td>
                                @if(!empty($account->detail_type))
                                    {{__('accounting::lang.' . $account->detail_type->name)}}
                                @endif
                            </td>
                        </tr>
                        <tr>
                            <th>@lang( 'lang_v1.balance' ):</th>
                            <td>@format_currency($current_bal)</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>

        <div class="col-md-7">
        
            <div class="box box-solid">
                <div class="box-header">
                    <h3 class="box-title"> <i class="fa fa-filter" aria-hidden="true"></i> @lang('report.filters'):</h3>
                </div>
                <div class="box-body">
                    <div class="col-sm-6">
                        <div class="form-group">
                            {!! Form::label('transaction_date_range', __('report.date_range') . ':') !!}
                            <div class="input-group">
                                <span class="input-group-addon"><i class="fa fa-calendar"></i></span>
                                {!! Form::text('transaction_date_range', null, ['class' => 'form-control', 'readonly', 'placeholder' => __('report.date_range')]) !!}
                            </div>
                        </div>
                    </div>

                    <div class="col-sm-6">
                        <div class="form-group">
                            {!! Form::label('all_accounts', __( 'accounting::lang.account' ) . ':') !!}
                            {!! Form::select('account_filter', [$account->id => $account->name], $account->id,
                                ['class' => 'form-control accounts-dropdown', 'style' => 'width:100%', 
                                'id' => 'account_filter', 'data-default' => $account->id]); !!}
                        </div>
                    </div>
                    
                </div>
            </div>

        </div>
    </div>
</section>

<section class="content">
    <div class="row">
        <div class="col-sm-12">
        	<div class="box">
                <div class="box-body">
                    @can('account.access')
                        <div class="table-responsive">
                    	<table class="table table-bordered table-striped" id="ledger">
                    		<thead>
                    			<tr>
                                    <th>@lang( 'messages.date' )</th>
                                    <th>@lang( 'lang_v1.description' )</th>
                                    <th>@lang( 'brand.note' )</th>
                                    <th>@lang( 'lang_v1.added_by' )</th>
                                    <th>@lang('account.debit')</th>
                                    <th>@lang('account.credit')</th>
                    				<!-- <th>@lang( 'lang_v1.balance' )</th> -->
                                    <th>@lang( 'messages.action' )</th>
                    			</tr>
                    		</thead>

                            

                            <tfoot>
                                <tr class="bg-gray font-17 footer-total text-center">
                                    <td colspan="4"><strong>@lang('sale.total'):</strong></td>
                                    <td class="footer_total_debit"></td>
                                    <td class="footer_total_credit"></td>
                                    <td></td>
                                </tr>
                            </tfoot>
                    	</table>
                        </div>
                    @endcan
                </div>
            </div>
        </div>
    </div>
</section>

@stop

@section('javascript')
@include('accounting::accounting.common_js')
<script>
    $(document).ready(function(){        
        $('#account_filter').change(function(){
            account_id = $(this).val();
            url = base_path + '/accounting/ledger/' + account_id;
            window.location = url;
        })

        dateRangeSettings.startDate = moment().subtract(6, 'days');
        dateRangeSettings.endDate = moment();
        $('#transaction_date_range').daterangepicker(
            dateRangeSettings,
            function (start, end) {
                $('#transaction_date_range').val(start.format(moment_date_format) + ' ~ ' + end.format(moment_date_format));
                
                ledger.ajax.reload();
            }
        );
        
        // Account Book
        ledger = $('#ledger').DataTable({
                            processing: true,
                            serverSide: true,
                            ajax: {
                                url: '{{action([\Modules\Accounting\Http\Controllers\CoaController::class, 'ledger'],[$account->id])}}',
                                data: function(d) {
                                    var start = '';
                                    var end = '';
                                    if($('#transaction_date_range').val()){
                                        start = $('input#transaction_date_range').data('daterangepicker').startDate.format('YYYY-MM-DD');
                                        end = $('input#transaction_date_range').data('daterangepicker').endDate.format('YYYY-MM-DD');
                                    }
                                    var transaction_type = $('select#transaction_type').val();
                                    d.start_date = start;
                                    d.end_date = end;
                                    d.type = transaction_type;
                                }
                            },
                            "ordering": false,
                            columns: [
                                {data: 'operation_date', name: 'operation_date'},
                                {data: 'ref_no', name: 'ATM.ref_no'},
                                {data: 'note', name: 'ATM.note'},
                                {data: 'added_by', name: 'added_by'},
                                {data: 'debit', name: 'amount', searchable: false},
                                {data: 'credit', name: 'amount', searchable: false},
                                //{data: 'balance', name: 'balance', searchable: false},
                                {data: 'action', name: 'action', searchable: false}
                            ],
                            "fnDrawCallback": function (oSettings) {
                                __currency_convert_recursively($('#ledger'));
                            },
                            "footerCallback": function ( row, data, start, end, display ) {
                                var footer_total_debit = 0;
                                var footer_total_credit = 0;

                                for (var r in data){
                                    footer_total_debit += $(data[r].debit).data('orig-value') ? parseFloat($(data[r].debit).data('orig-value')) : 0;
                                    footer_total_credit += $(data[r].credit).data('orig-value') ? parseFloat($(data[r].credit).data('orig-value')) : 0;
                                }

                                $('.footer_total_debit').html(__currency_trans_from_en(footer_total_debit));
                                $('.footer_total_credit').html(__currency_trans_from_en(footer_total_credit));
                            }
                        });
        $('#transaction_date_range').on('cancel.daterangepicker', function(ev, picker) {
            $('#transaction_date_range').val('');
            ledger.ajax.reload();
        });
    });
</script>
@stop