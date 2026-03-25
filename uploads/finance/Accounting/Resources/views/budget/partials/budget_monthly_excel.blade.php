<table>
    <thead>
        <tr>
            <th>
                @lang('account.account')
            </th>
            @foreach($months as $k => $m)
                <th>{{Carbon::createFromFormat('m', $k)->format('M')}}</th>
            @endforeach
            <th>@lang('sale.total')</th>
        </tr>
    </thead>
    <tbody>
    @foreach($account_types as $account_type => $account_type_detail )
        <tr>
            <th colspan="14">
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
                    <th>{{$account->name}}</th>
                    @foreach($months as $k => $m)
                        @php
                            $account_budget = $budget->where('accounting_account_id', $account->id)->first();
                            $value = !is_null($account_budget) && !is_null($account_budget->$m) 
                            ? $account_budget->$m : null;
                        @endphp
                        <td>
                            @if(!is_null($value))
                                {{@num_format($value)}}  
                                @php
                                    $total += $value;   
                                @endphp                                           
                            @endif
                        </td>
                    @endforeach
                    <td>
                        {{@num_format($total)}} 
                    </td>
                </tr>
            @endforeach
            <tr>
                <th>
                    @lang('sale.total')
                </th>
                @foreach($months as $k => $m)
                    <td>{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum($m))}}</td>
                @endforeach
                <td></td>
            </tr>
    @endforeach
    <tr>
        <th>
            @lang('lang_v1.grand_total')
        </th>
        @foreach($months as $k => $m)
            <td>{{@num_format($budget->sum($m))}}</td>
        @endforeach
        <td></td>
    </tr>
    </tbody>
</table>