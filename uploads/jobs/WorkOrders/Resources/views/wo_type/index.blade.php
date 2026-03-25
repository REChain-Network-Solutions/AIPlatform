@extends('layouts.app')
@section('page-title')
    {{ __('WO Type') }}
@endsection
@section('breadcrumb')
    <ul class="breadcrumb mb-0">
        <li class="breadcrumb-item">
            <a href="{{ route('dashboard') }}">{{ __('Dashboard') }}</a>
        </li>
        <li class="breadcrumb-item active">
            <a href="#">
                {{ __('WO Type') }}
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
                            <h5>{{ __('WO Type List') }}</h5>
                        </div>
                        <div class="col-auto">
                            @if (Gate::check('create wo type'))
                                <a class="btn btn-secondary customModal" href="#" data-size="lg"
                                    data-url="{{ route('wo-type.create') }}" data-title="{{ __('Create Type') }}">
                                    <i class="ti ti-circle-plus align-text-bottom"></i>
                                    {{ __('Create Type') }}
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
                                    <th>{{ __('Type') }}</th>
                                    <th>{{ __('Action') }}</th>
                                </tr>
                            </thead>
                            <tbody>
                                @foreach ($woTypes as $woType)
                                    <tr>
                                        <td>{{ $woType->type }} </td>
                                        <td>
                                            <div class="cart-action">
                                                {!! Form::open(['method' => 'DELETE', 'route' => ['wo-type.destroy', $woType->id]]) !!}
                                                @can('edit wo type')
                                                    <a class="avtar avtar-xs btn-link-secondary text-secondary customModal"
                                                        data-bs-toggle="tooltip" data-size="md"
                                                        data-bs-original-title="{{ __('Edit') }}" href="#"
                                                        data-url="{{ route('wo-type.edit', $woType->id) }}"
                                                        data-title="{{ __('Edit Asset') }}"> <i data-feather="edit"></i></a>
                                                @endcan
                                                @can('delete wo type')
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
