@extends('layouts.app')
@section('page-title')
    {{ __('Work Order') }}
@endsection
@section('breadcrumb')
    <ul class="breadcrumb mb-0">
        <li class="breadcrumb-item">
            <a href="{{ route('dashboard') }}">{{ __('Dashboard') }}</a>
        </li>
        <li class="breadcrumb-item active">
            <a href="#">
                {{ __('Work Order') }}
            </a>
        </li>
    </ul>
@endsection
 
@section('content')
    <div class="row">
        <div class="col-12">
            <div class="card">
                <div class="card-header">
                    <div class="row align-items-center g-2">
                        <div class="col">
                            <h5>{{ __('Workorder List') }}</h5>
                        </div>
                        <div class="col-auto">
                            @if (Gate::check('create work order'))
                                <a class="btn btn-secondary" href="{{ route('workorder.create') }}">
                                    <i class="ti ti-circle-plus align-text-bottom"></i>
                                    {{ __('Create Workorder') }}
                                </a>
                            @endif
                        </div>
                    </div>
                </div>
                <div class="card-body pt-0">
                    <div class="dt-responsive table-responsive">
                        <table class="table table-hover advance-datatable">
                            <thead>
                                <tr>
                                    <th>{{ __('ID') }}</th>
                                    <th>{{ __('Detail') }}</th>
                                    <th>{{ __('Type') }}</th>
                                    <th>{{ __('Client') }}</th>
                                    <th>{{ __('Asset') }}</th>
                                    <th>{{ __('Due Date') }}</th>
                                    <th>{{ __('Priority') }}</th>
                                    <th>{{ __('Status') }}</th>
                                    <th>{{ __('Assign') }}</th>
                                    <th>{{ __('Total') }}</th>
                                    <th>{{ __('Action') }}</th>
                                </tr>
                            </thead>
                            <tbody>
                                @foreach ($workorders as $workorder)
                                    <tr>
                                        <td>{{ workOrderPrefix() . $workorder->wo_id }} </td>
                                        <td>{{ $workorder->wo_detail }} </td>
                                        <td>{{ !empty($workorder->types) ? $workorder->types->type : '-' }} </td>
                                        <td>{{ !empty($workorder->clients) ? $workorder->clients->name : '-' }} </td>
                                        <td>{{ !empty($workorder->assets) ? $workorder->assets->name : '-' }} </td>
                                        <td>{{ dateFormat($workorder->due_date) }} </td>
                                        <td>
                                            @if ($workorder->priority == 'low')
                                                <span
                                                    class="badge text-bg-primary">{{ \App\Models\WORequest::$priority[$workorder->priority] }}</span>
                                            @elseif($workorder->priority == 'medium')
                                                <span
                                                    class="badge text-bg-info">{{ \App\Models\WORequest::$priority[$workorder->priority] }}</span>
                                            @elseif($workorder->priority == 'high')
                                                <span
                                                    class="badge text-bg-warning">{{ \App\Models\WORequest::$priority[$workorder->priority] }}</span>
                                            @elseif($workorder->priority == 'critical')
                                                <span
                                                    class="badge text-bg-danger">{{ \App\Models\WORequest::$priority[$workorder->priority] }}</span>
                                            @endif
                                        </td>
                                        <td>
                                            @if ($workorder->status == 'pending')
                                                <span
                                                    class="badge text-bg-warning">{{ \App\Models\Workorder::$status[$workorder->status] }}</span>
                                            @elseif($workorder->status == 'on_hold')
                                                <span
                                                    class="badge text-bg-primary">{{ \App\Models\Workorder::$status[$workorder->status] }}</span>
                                            @elseif($workorder->status == 'approved' || $workorder->status == 'completed')
                                                <span
                                                    class="badge text-bg-success">{{ \App\Models\Workorder::$status[$workorder->status] }}</span>
                                            @else
                                                <span
                                                    class="badge text-bg-danger">{{ \App\Models\Workorder::$status[$workorder->status] }}</span>
                                            @endif

                                        </td>
                                        <td>{{ !empty($workorder->assigned) ? $workorder->assigned->name : '-' }} </td>
                                        <td>{{ priceFormat($workorder->getWorkorderTotalAmount()) }}</td>
                                        <td>
                                            <div class="cart-action">
                                                {!! Form::open(['method' => 'DELETE', 'route' => ['workorder.destroy', $workorder->id]]) !!}
                                                @can('show work order')
                                                    <a class="avtar avtar-xs btn-link-warning text-warning"
                                                        data-bs-toggle="tooltip"
                                                        href="{{ route('workorder.show', \Illuminate\Support\Facades\Crypt::encrypt($workorder->id)) }}"
                                                        data-title="{{ __('Workorder Detail') }}"> <i
                                                            data-feather="eye"></i></a>
                                                @endcan
                                                @can('edit work order')
                                                    <a class="avtar avtar-xs btn-link-secondary text-secondary"
                                                        data-bs-toggle="tooltip"
                                                        href="{{ route('workorder.edit', \Illuminate\Support\Facades\Crypt::encrypt($workorder->id)) }}"
                                                        data-title="{{ __('Edit Workorder') }}"> <i
                                                            data-feather="edit"></i></a>
                                                @endcan
                                                @can('delete work order')
                                                    <a class=" avtar avtar-xs btn-link-danger text-danger confirm_dialog"
                                                        data-bs-toggle="tooltip" data-bs-original-title="{{ __('Detete') }}"
                                                        href="#"> <i data-feather="trash-2"></i></a>
                                                @endcan
                                                {!! Form::close() !!}
                                            </div>

                                        </td>
                                    </tr>
                                @endforeach

                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
@endsection
