<table>
    <thead>
        <tr>
            <th>
                @lang('account.account')
            </th>
            <th>
            {{$fy_year}}
            </th>
        </tr>
    </thead>
    @foreach($account_types as $account_type => $account_type_detail )
    <tbody>
        <tr>
            <th colspan="2">
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
                <th>{{$account->name}}</th>
                <td>
                    @if(!is_null($account_budget) && !is_null($account_budget->yearly)) 
                        {{@num_format($account_budget->yearly)}}
                    @else 
                        0
                    @endif
                </td>
            </tr>
        @endforeach
        <tr>
            <th>
                @lang('sale.total')
            </th>
            <td>{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum('yearly'))}}</td>
        </tr>
    </tbody>
    @endforeach
    <tfoot>
        <tr>
            <th>
                @lang('lang_v1.grand_total')
            </th>
            <td>{{@num_format($budget->sum('yearly'))}}</td>
        </tr>
    </tfoot>
</table>