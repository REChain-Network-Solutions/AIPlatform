<link rel="stylesheet" href="{{ asset('css/app.css?v='.$asset_v) }}">
<style>
    .header {
        background-color: #dadada;
    }
    .header td, .header th {
        padding: 10px;
    }
</style>
<div class="width-100">
    <h1>{{__('accounting::lang.budget')}}</h1>
    <h4>@lang( 'accounting::lang.financial_year_for_the_budget' ): {{$fy_year}}</h4>
</div>
<div class="width-100">
    <table class="table table-striped width-100">
        <thead>
            <tr class="header">
                <th align="left">
                    @lang('account.account')
                </th>
                <th align="left">
                    @lang('accounting::lang.1st_quarter')
                </th>
                <th align="left">
                    @lang('accounting::lang.2nd_quarter')
                </th>
                <th align="left">
                    @lang('accounting::lang.3rd_quarter')
                </th>
                <th align="left">
                    @lang('accounting::lang.4th_quarter')
                </th>
                <th align="left">@lang('sale.total')</th>
            </tr>
        </thead>
        @foreach($account_types as $account_type => $account_type_detail )
        <tbody>
            <tr class="header" >
                <th colspan="6" align="center">
                    {{$account_type_detail['label']}}
                </th>
            </tr>
            @php
                $account_ids=[];
            @endphp
            @foreach($accounts->where('account_primary_type', $account_type)->sortBy('name')->all() as $account)
                @php
                    $total = 0;
                    $account_ids[]=$account->id;
                    $account_budget = $budget->where('accounting_account_id', $account->id)->first();
                @endphp
                    <tr>
                        <th align="left">{{$account->name}}</th>
                        <td align="left">
                            @if(!is_null($account_budget) && !is_null($account_budget->quarter_1))
                                @php
                                    $total += $account_budget->quarter_1;   
                                @endphp  
                                {{@num_format($account_budget->quarter_1)}}
                            @else
                                0
                            @endif
                        </td>
                        <td align="left">
                            @if(!is_null($account_budget) && !is_null($account_budget->quarter_2)) 
                                @php
                                    $total += $account_budget->quarter_2;   
                                @endphp
                                {{@num_format($account_budget->quarter_2)}}
                            @else
                                0
                            @endif
                        </td>
                        <td align="left">
                            @if(!is_null($account_budget) && !is_null($account_budget->quarter_3)) 
                                @php
                                    $total += $account_budget->quarter_3;   
                                @endphp
                                {{@num_format($account_budget->quarter_2)}}
                            @else
                                0
                            @endif
                        </td>
                        <td align="left">
                            @if(!is_null($account_budget) && !is_null($account_budget->quarter_4))
                                @php
                                    $total += $account_budget->quarter_4;   
                                @endphp 
                                {{@num_format($account_budget->quarter_4)}}
                            @else
                                0
                            @endif
                        </td>
                        <td align="left">
                            {{@num_format($total)}} 
                        </td>
                    </tr>
            @endforeach
                <tr class="header">
                    <th align="left">
                        @lang('sale.total')
                    </th>
                    <td align="left">{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum('quarter_1'))}}</td>
                    <td align="left">{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum('quarter_2'))}}</td>
                    <td align="left">{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum('quarter_3'))}}</td>
                    <td align="left">{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum('quarter_4'))}}</td>
                    <td align="left"></td>
                </tr>
            </tbody>
        @endforeach
        <tfoot>
            <tr class="header">
                <th align="left">
                    @lang('lang_v1.grand_total')
                </th>
                <td align="left">{{@num_format($budget->sum('quarter_1'))}}</td>
                <td align="left">{{@num_format($budget->sum('quarter_2'))}}</td>
                <td align="left">{{@num_format($budget->sum('quarter_3'))}}</td>
                <td align="left">{{@num_format($budget->sum('quarter_4'))}}</td>
                <td></td>
            </tr>
        </tfoot>
    </table>
</div>