<table>
    <thead>
        <tr>
            <th>
                @lang('account.account')
            </th>
            <th>
                @lang('accounting::lang.1st_quarter')
            </th>
            <th>
                @lang('accounting::lang.2nd_quarter')
            </th>
            <th>
                @lang('accounting::lang.3rd_quarter')
            </th>
            <th>
                @lang('accounting::lang.4th_quarter')
            </th>
            <th>@lang('sale.total')</th>
        </tr>
    </thead>
    @foreach($account_types as $account_type => $account_type_detail )
    <tbody>
        <tr>
            <th colspan="6">
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
                    <th>{{$account->name}}</th>
                    <td>
                        @if(!is_null($account_budget) && !is_null($account_budget->quarter_1))
                            @php
                                $total += $account_budget->quarter_1;   
                            @endphp  
                            {{@num_format($account_budget->quarter_1)}}
                        @else
                            0
                        @endif
                    </td>
                    <td>
                        @if(!is_null($account_budget) && !is_null($account_budget->quarter_2)) 
                            @php
                                $total += $account_budget->quarter_2;   
                            @endphp
                            {{@num_format($account_budget->quarter_2)}}
                        @else
                            0
                        @endif
                    </td>
                    <td>
                        @if(!is_null($account_budget) && !is_null($account_budget->quarter_3)) 
                            @php
                                $total += $account_budget->quarter_3;   
                            @endphp
                            {{@num_format($account_budget->quarter_2)}}
                        @else
                            0
                        @endif
                    </td>
                    <td>
                        @if(!is_null($account_budget) && !is_null($account_budget->quarter_4))
                            @php
                                $total += $account_budget->quarter_4;   
                            @endphp 
                            {{@num_format($account_budget->quarter_4)}}
                        @else
                            0
                        @endif
                    </td>
                    <td>
                        {{@num_format($total)}} 
                    </td>
                </tr>
        @endforeach
            <tr>
                <th>
                    @lang('sale.total')
                </th>
                <td>{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum('quarter_1'))}}</td>
                <td>{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum('quarter_2'))}}</td>
                <td>{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum('quarter_3'))}}</td>
                <td>{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum('quarter_4'))}}</td>
                <td></td>
            </tr>
        </tbody>
    @endforeach
    <tfoot>
        <tr>
            <th>
                @lang('lang_v1.grand_total')
            </th>
            <td>{{@num_format($budget->sum('quarter_1'))}}</td>
            <td>{{@num_format($budget->sum('quarter_2'))}}</td>
            <td>{{@num_format($budget->sum('quarter_3'))}}</td>
            <td>{{@num_format($budget->sum('quarter_4'))}}</td>
            <td></td>
        </tr>
    </tfoot>
</table>