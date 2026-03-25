@extends('layouts.main')
@section('page-title')
    {{ __('Manage Project Report') }}
@endsection
@section('page-breadcrumb')
    {{ __('Project Report') }}
@endsection
@push('css')
    @include('layouts.includes.datatable-css')
@endpush
@php
    $client_keyword = Auth::user()->hasRole('client') ? 'client.' : '';
@endphp

@section('content')
    <div class="row">
        <div id="multiCollapseExample1">
            <div class="card">
                <div class="card-body">
                    <div class="row d-flex align-items-center justify-content-end">
                        @if (Auth::user()->hasRole('company') || Auth::user()->hasRole('client'))
                            <div class="col-2 form-group">
                                {{ Form::label('user', __('User'), ['class' => 'form-label']) }}
                                {{ Form::select('user', $users, isset($_GET['user']) ? $_GET['user'] : '', ['class' => 'form-control ', 'placeholder' => 'All Users']) }}
                            </div>
                        @endif
                        <div class="col-2 form-group">
                                {{ Form::label('status', __('All Status'), ['class' => 'form-label']) }}
                                <select class="form-control" name="status" id="status">
                                    <option value="" class="px-4">{{ __('All Status') }}</option>
                                    <option value="Ongoing">{{ __('Ongoing') }}</option>
                                    <option value="Finished">{{ __('Finished') }}</option>
                                    <option value="OnHold">{{ __('OnHold') }}</option>
                                </select>
                        </div>
                        <div class="form-group col-md-3">
                            {{ Form::label('start_date', __('Start Date'), ['class' => 'form-label']) }}
                            {{ Form::date('start_date', isset($_GET['start_date']) ? $_GET['start_date'] : null, ['class' => 'form-control ','placeholder' => 'Select Date']) }}
                        </div>
                        <div class="form-group col-md-3">
                            {{ Form::label('end_date', __('End Date'), ['class' => 'form-label']) }}
                            {{ Form::date('end_date', isset($_GET['end_date']) ? $_GET['end_date'] : null, ['class' => 'form-control ','placeholder' => 'Select Date']) }}
                        </div>
                        <div class="col-auto d-flex">
                            <a  class="btn btn-sm btn-primary me-2"
                                data-bs-toggle="tooltip" title="{{ __('Apply') }}" id="applyfilter"
                                data-original-title="{{ __('apply') }}">
                                <span class="btn-inner--icon"><i class="ti ti-search"></i></span>
                            </a>
                            <a href="#!" class="btn btn-sm btn-danger "
                                data-bs-toggle="tooltip" title="{{ __('Reset') }}" id="clearfilter"
                                data-original-title="{{ __('Reset') }}">
                                <span class="btn-inner--icon"><i class="ti ti-trash-off text-white-off "></i></span>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="col-xl-12">
            <div class="card">
                <div class="card-body table-border-style">
                    <div class="table-responsive overflow_hidden">
                        {{ $dataTable->table(['width' => '100%']) }}
                    </div>
                </div>
            </div>
        </div>
    </div>
    </div>



@endsection
@push('scripts')
    @include('layouts.includes.datatable-js')
    {{ $dataTable->scripts() }}
@endpush
