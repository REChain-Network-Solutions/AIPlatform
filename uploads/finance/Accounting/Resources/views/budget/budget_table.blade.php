<div class="row">
    <div class="col-md-12">
        <div class="nav-tabs-custom">
            <ul class="nav nav-tabs nav-justified">
                <li class="active">
                    <a href="#monthly_tab" data-toggle="tab" 
                    aria-expanded="true">@lang('accounting::lang.monthly')</a>
                </li>
                <li>
                    <a href="#quarterly_tab" data-toggle="tab" 
                    aria-expanded="true">@lang('accounting::lang.quarterly')</a>
                </li>
                <li>
                    <a href="#yearly_tab" data-toggle="tab" 
                    aria-expanded="true">@lang('accounting::lang.yearly')</a>
                </li>
            </ul>
            <div class="tab-content">
                <div class="tab-pane active" id="monthly_tab">
                    <div class="text-right mb-12">
                        <a class="btn btn-sm btn-default" 
                        href="{{route('budget.index')}}?financial_year={{$fy_year}}&format=pdf&view_type=monthly"><i class="fas fa-file-pdf"></i> 
                            @lang('accounting::lang.export_to_pdf')</a>
                        <a class="btn btn-sm btn-default" 
                        href="{{route('budget.index')}}?financial_year={{$fy_year}}&format=csv&view_type=monthly"><i class="fas fa-file-csv"></i> 
                            @lang('accounting::lang.export_to_csv')</a>
                        <a class="btn btn-sm btn-default" 
                        href="{{route('budget.index')}}?financial_year={{$fy_year}}&format=excel&view_type=monthly"><i class="fas fa-file-excel"></i> 
                            @lang('accounting::lang.export_to_excel')</a>
                    </div>
                    <div class="table-responsive" style="height: 500px;">
                    <table class="table table-striped table-sticky">
                        <thead>
                            <tr class="bg-green">
                                <th>
                                    @lang('account.account')
                                </th>
                                @foreach($months as $k => $m)
                                    <th>{{Carbon::createFromFormat('m', $k)->format('M')}}</th>
                                @endforeach
                                <th>@lang('sale.total')</th>
                            </tr>
                        </thead>
                        @foreach($account_types as $account_type => $account_type_detail )
                        <tbody class="collapsed">
                            <tr class="toggle-tr bg-gray" style="cursor: pointer;">
                                <th colspan="14">
                                    <span class="collapse-icon">
                                        <i class="fas fa-arrow-circle-right"></i>
                                    </span>
                                    {{$account_type_detail['label']}}
                                </th>
                            </tr>
                                @php
                                    $account_ids=[];
                                @endphp
                                @foreach($accounts->where('account_primary_type', $account_type)->sortBy('name')->all() as $account)
                                    <tr class="collapse-tr">
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
                                <tr class="collapse-tr bg-gray">
                                    <th>
                                        @lang('sale.total')
                                    </th>
                                    @foreach($months as $k => $m)
                                        <td>{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum($m))}}</td>
                                    @endforeach
                                    <td></td>
                                </tr>
                            </tbody>
                        @endforeach
                        <tfoot class="bg-green">
                            <tr class="table-footer">
                                <th>
                                    @lang('lang_v1.grand_total')
                                </th>
                                @foreach($months as $k => $m)
                                    <td>{{@num_format($budget->sum($m))}}</td>
                                @endforeach
                                <td></td>
                            </tr>
                        </tfoot>
                    </table>
                    </div>
                </div>
                <div class="tab-pane" id="quarterly_tab">
                    <div class="text-right mb-12">
                        <a class="btn btn-sm btn-default" 
                        href="{{route('budget.index')}}?financial_year={{$fy_year}}&format=pdf&view_type=quarterly"><i class="fas fa-file-pdf"></i> 
                            @lang('accounting::lang.export_to_pdf')</a>
                        <a class="btn btn-sm btn-default" 
                        href="{{route('budget.index')}}?financial_year={{$fy_year}}&format=csv&view_type=quarterly"><i class="fas fa-file-csv"></i> 
                            @lang('accounting::lang.export_to_csv')</a>
                        <a class="btn btn-sm btn-default" 
                        href="{{route('budget.index')}}?financial_year={{$fy_year}}&format=excel&view_type=quarterly"><i class="fas fa-file-excel"></i> 
                            @lang('accounting::lang.export_to_excel')</a>
                    </div>
                    <div class="table-responsive" style="height: 500px;">
                        <table class="table table-striped table-sticky">
                            <thead>
                                <tr class="bg-green">
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
                            <tbody class="collapsed">
                                <tr class="toggle-tr bg-gray" style="cursor: pointer;">
                                    <th colspan="6">
                                        <span class="collapse-icon">
                                            <i class="fas fa-arrow-circle-right"></i>
                                        </span>
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
                                        <tr class="collapse-tr">
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
                                    <tr class="collapse-tr bg-gray">
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
                                <tr class="bg-green">
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
                    </div>
                </div>
                <div class="tab-pane" id="yearly_tab">
                    <div class="text-right mb-12">
                        <a class="btn btn-sm btn-default" 
                        href="{{route('budget.index')}}?financial_year={{$fy_year}}&format=pdf&view_type=yearly"><i class="fas fa-file-pdf"></i> 
                            @lang('accounting::lang.export_to_pdf')</a>
                        <a class="btn btn-sm btn-default" 
                        href="{{route('budget.index')}}?financial_year={{$fy_year}}&format=csv&view_type=yearly"><i class="fas fa-file-csv"></i> 
                            @lang('accounting::lang.export_to_csv')</a>
                        <a class="btn btn-sm btn-default" 
                        href="{{route('budget.index')}}?financial_year={{$fy_year}}&format=excel&view_type=yearly"><i class="fas fa-file-excel"></i> 
                            @lang('accounting::lang.export_to_excel')</a>
                    </div>
                    <div class="table-responsive" style="height: 500px;">
                        <table class="table table-striped table-sticky">
                            <thead>
                                <tr class="bg-green">
                                    <th>
                                        @lang('account.account')
                                    </th>
                                    <th class="text-center">
                                    {{$fy_year}}
                                    </th>
                                </tr>
                            </thead>
                            @foreach($account_types as $account_type => $account_type_detail )
                            <tbody class="collapsed">
                                <tr class="toggle-tr bg-gray" style="cursor: pointer;">
                                    <th colspan="2">
                                        <span class="collapse-icon">
                                            <i class="fas fa-arrow-circle-right"></i>
                                        </span>
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
                                    <tr class="collapse-tr">
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
                                <tr class="bg-gray collapse-tr">
                                    <th>
                                        @lang('sale.total')
                                    </th>
                                    <td>{{@num_format($budget->whereIn('accounting_account_id', $account_ids)->sum('yearly'))}}</td>
                                </tr>
                            </tbody>
                            @endforeach
                            <tfoot>
                                <tr class="bg-green">
                                    <th>
                                        @lang('lang_v1.grand_total')
                                    </th>
                                    <td>{{@num_format($budget->sum('yearly'))}}</td>
                                </tr>
                            </tfoot>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>