@extends('layouts.app')

@section('title', __('accounting::lang.budget_for_fy', ['fy' => $fy_year]))

@section('content')

@include('accounting::layouts.nav')

<!-- Content Header (Page header) -->
<section class="content-header">
    <h1 class="tw-text-xl md:tw-text-3xl tw-font-bold tw-text-black">@lang('accounting::lang.budget_for_fy', ['fy' => $fy_year])</h1>
</section>
<section class="content">
	@component('components.widget', ['class' => 'box-solid'])
    {!! Form::open(['url' => action([\Modules\Accounting\Http\Controllers\BudgetController::class, 'store']), 
            'method' => 'post', 'id' => 'add_budget_form' ]) !!}
        <input type="hidden" name="financial_year" value="{{$fy_year}}">
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
                            <div class="table-responsive" style="height: 500px;">
                                <table class="table table-striped">
                                    <tr>
                                        <th>
                                            @lang('account.account')
                                        </th>
                                        @foreach($months as $k => $m)
                                            <th>{{Carbon::createFromFormat('m', $k)->format('M')}}</th>
                                        @endforeach
                                    </tr>
                                    @foreach($accounts as $account)
                                        <tr>
                                            <th>{{$account->name}}</th>
                                            @foreach($months as $k => $m)
                                                @php
                                                    $account_budget = $budget->where('accounting_account_id', $account->id)->first();
                                                    $value = !is_null($account_budget) && !is_null($account_budget->$m) 
                                                    ? $account_budget->$m : null;
                                                @endphp
                                                <td>
                                                    <input type="text" class="form-control input_number" 
                                                    name="budget[{{$account->id}}][{{$m}}]" @if(!is_null($value))
                                                    value="{{@num_format($value)}}" @endif>
                                                </td>
                                            @endforeach
                                        </tr>
                                    @endforeach
                                </table>
                            </div>
                        </div>
                        <div class="tab-pane" id="quarterly_tab">
                            <div class="table-responsive" style="height: 500px;">
                                <table class="table table-striped">
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
                                    </tr>
                                    @foreach($accounts as $account)
                                            @php
                                                $account_budget = $budget->where('accounting_account_id', $account->id)->first();
                                            @endphp
                                        <tr>
                                            <th>{{$account->name}}</th>
                                            <td>
                                                <input type="text" class="form-control input_number" 
                                                name="budget[{{$account->id}}][quarter_1]"
                                                @if(!is_null($account_budget) && !is_null($account_budget->quarter_1)) 
                                                value="{{@num_format($account_budget->quarter_1)}}" @endif >
                                            </td>
                                            <td>
                                                <input type="text" class="form-control input_number" 
                                                name="budget[{{$account->id}}][quarter_2]"
                                                @if(!is_null($account_budget) && !is_null($account_budget->quarter_2)) 
                                                value="{{@num_format($account_budget->quarter_2)}}" @endif
                                                >
                                            </td>
                                            <td>
                                                <input type="text" class="form-control input_number" 
                                                name="budget[{{$account->id}}][quarter_3]"
                                                @if(!is_null($account_budget) && !is_null($account_budget->quarter_3)) 
                                                value="{{@num_format($account_budget->quarter_3)}}" @endif
                                                >
                                            </td>
                                            <td>
                                                <input type="text" class="form-control input_number" 
                                                name="budget[{{$account->id}}][quarter_4]"
                                                @if(!is_null($account_budget) && !is_null($account_budget->quarter_4)) 
                                                value="{{@num_format($account_budget->quarter_4)}}" @endif
                                                >
                                            </td>
                                        </tr>
                                    @endforeach
                                </table>
                            </div>
                        </div>
                        <div class="tab-pane" id="yearly_tab">
                            <div class="table-responsive" style="height: 500px;">
                                <table class="table table-striped">
                                    <tr>
                                        <th>
                                            @lang('account.account')
                                        </th>
                                        <th class="text-center">
                                        {{$fy_year}}
                                        </th>
                                    </tr>
                                    @foreach($accounts as $account)
                                        @php
                                            $account_budget = $budget->where('accounting_account_id', $account->id)->first();
                                        @endphp
                                        <tr>
                                            <th>{{$account->name}}</th>
                                            <td>
                                                <input type="text" class="form-control input_number" 
                                                name="budget[{{$account->id}}][yearly]"
                                                @if(!is_null($account_budget) && !is_null($account_budget->yearly)) 
                                                value="{{@num_format($account_budget->yearly)}}" @endif
                                                >
                                            </td>
                                        </tr>
                                    @endforeach
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-md-12 text-center">
                <button type="submit" class="tw-dw-btn tw-dw-btn-primary tw-text-white tw-dw-btn-lg">@lang('messages.submit')</button>
            </div>
        </div>
    {!! Form::close() !!}
    @endcomponent
</section>
@stop
@section('javascript')
<script type="text/javascript">
	$(document).ready( function(){
	});
</script>
@endsection