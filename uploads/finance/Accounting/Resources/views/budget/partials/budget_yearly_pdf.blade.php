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
                {{$fy_year}}
                </th>
            </tr>
        </thead>
        @foreach($account_types as $account_type => $account_type_detail )
        <tbody>
            <tr class="header" style="cursor: pointer;">
                <th colspan="2" align="center">
                    {{$account_type_detail['label']}}
                </th>
            </tr>
            @php
                $account_ids=[];
            @endphp
            @foreach($accounts->where('account_primary_type', $account_type)->sortBy('name')->all() as $account)
                @php
                    $account_ids[]=$account->id;
                    $account_budget = $budget->where('accounting_account_id', $account->id)->first();
                @endphp
                <tr>
                    <th align="left">{{$account->name}}</th>
                    <td align="left">
                        @if(!is_null($account_budget) && !is_null($account_budget->yearly)) 
                            {{@num_format($account_budget->yearly)}}
                        @else 
                            0
                        @endif
                    </td>
                </tr>
            @endforeach
            <tr class="header">
                <th align="left">
                    @lang('sale.total')
                </th>
                <td align="left">{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum('yearly'))}}</td>
            </tr>
        </tbody>
        @endforeach
        <tfoot>
            <tr class="header">
                <th align="left">
                    @lang('lang_v1.grand_total')
                </th>
                <td align="left">{{@num_format($budget->sum('yearly'))}}</td>
            </tr>
        </tfoot>
    </table>
</div>