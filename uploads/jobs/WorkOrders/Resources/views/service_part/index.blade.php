@extends('layouts.app')
@section('page-title')
    {{ __('Services & Parts') }}
@endsection
@section('breadcrumb')
    <ul class="breadcrumb mb-0">
        <li class="breadcrumb-item">
            <a href="{{ route('dashboard') }}">{{ __('Dashboard') }}</a>
        </li>
        <li class="breadcrumb-item active">
            <a href="#">
                {{ __('Services & Parts') }}
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
                            <h5>{{ __('Service & Part List') }}</h5>
                        </div>
                        <div class="col-auto">
                            @if (Gate::check('create service & part'))
                                <a class="btn btn-secondary customModal" href="#" data-size="lg"
                                    data-url="{{ route('services-parts.create') }}"
                                    data-title="{{ __('Create Service & Part') }}"> <i
                                        class="ti ti-circle-plus align-text-bottom"></i>
                                    {{ __('Create Service & Part') }}
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
                                    <th>{{ __('Title') }}</th>
                                    <th>{{ __('SKU') }}</th>
                                    <th>{{ __('Price') }}</th>
                                    <th>{{ __('Unit') }}</th>
                                    <th>{{ __('Type') }}</th>
                                    <th>{{ __('Description') }}</th>
                                    <th>{{ __('Action') }}</th>
                                </tr>
                            </thead>
                            <tbody>
                                @foreach ($serviceParts as $servicePart)
                                    <tr>
                                        <td>{{ $servicePart->title }} </td>
                                        <td>{{ $servicePart->sku }} </td>
                                        <td>{{ priceFormat($servicePart->price) }} </td>
                                        <td>{{ $servicePart->unit }} </td>
                                        <td>{{ ucfirst($servicePart->type) }} </td>
                                        <td>{{ !empty($servicePart->description) ? $servicePart->description : '-' }} </td>
                                        <td>
                                            <div class="cart-action">
                                                {!! Form::open(['method' => 'DELETE', 'route' => ['services-parts.destroy', $servicePart->id]]) !!}
                                                @can('show service & part')
                                                    <a class="avtar avtar-xs btn-link-warning text-warning customModal" data-bs-toggle="tooltip" data-size="lg"
                                                        data-bs-original-title="{{ __('Details') }}" href="#"
                                                        data-url="{{ route('services-parts.show', $servicePart->id) }}"
                                                        data-title="{{ __('Service & Part Detail') }}"> <i
                                                            data-feather="eye"></i></a>
                                                @endcan
                                                @can('edit service & part')
                                                    <a class="avtar avtar-xs btn-link-secondary text-secondary customModal"
                                                        data-bs-toggle="tooltip" data-size="lg"
                                                        data-bs-original-title="{{ __('Edit') }}" href="#"
                                                        data-url="{{ route('services-parts.edit', $servicePart->id) }}"
                                                        data-title="{{ __('Edit Service & Part') }}"> <i
                                                            data-feather="edit"></i></a>
                                                @endcan
                                                @can('delete service & part')
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
