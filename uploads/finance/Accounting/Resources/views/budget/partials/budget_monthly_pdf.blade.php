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
    <table class="table table-striped width-100" style="text-align: left;">
        <thead>
            <tr class="header">
                <th align="left">
                    @lang('account.account')
                </th>
                @foreach($months as $k => $m)
                    <th align="left">{{Carbon::createFromFormat('m', $k)->format('M')}}</th>
                @endforeach
                <th align="left">@lang('sale.total')</th>
            </tr>
        </thead>
        @foreach($account_types as $account_type => $account_type_detail )
        <tbody>
            <tr class="header">
                <th colspan="14" align="center">
                    {{$account_type_detail['label']}}
                </th>
            </tr>
                @php
                    $account_ids=[];
                @endphp
                @foreach($accounts->where('account_primary_type', $account_type)->sortBy('name')->all() as $account)
                    <tr>
                        @php
                            $total = 0;
                            $account_ids[]=$account->id;
                        @endphp
                        <th align="left">{{$account->name}}</th>
                        @foreach($months as $k => $m)
                            @php
                                $account_budget = $budget->where('accounting_account_id', $account->id)->first();
                                $value = !is_null($account_budget) && !is_null($account_budget->$m) 
                                ? $account_budget->$m : null;
                            @endphp
                            <td align="left">
                                @if(!is_null($value))
                                    {{@num_format($value)}}  
                                    @php
                                        $total += $value;   
                                    @endphp                                           
                                @endif
                            </td>
                        @endforeach
                        <td align="left">
                            {{@num_format($total)}} 
                        </td>
                    </tr>
                @endforeach
                <tr class="header">
                    <th align="left">
                        @lang('sale.total')
                    </th align="left">
                    @foreach($months as $k => $m)
                        <td>{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum($m))}}</td>
                    @endforeach
                    <td></td>
                </tr>
            </tbody>
        @endforeach
        <tfoot class="bg-green">
            <tr class="header">
                <th align="left">
                    @lang('lang_v1.grand_total')
                </th>
                @foreach($months as $k => $m)
                    <td align="left">{{@num_format($budget->sum($m))}}</td>
                @endforeach
                <td></td>
            </tr>
        </tfoot>
    </table>
</div>