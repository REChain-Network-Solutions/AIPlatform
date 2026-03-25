@extends('layouts.main')
@section('page-title')
    {{__('Manage POS Order')}}
@endsection
@section('page-breadcrumb')
   {{__('POS Order')}}
@endsection
@push('css')
@include('layouts.includes.datatable-css')
@endpush

@push('scripts')
@endpush

@section('page-action')
<div class="d-flex">
    @stack('addButtonHook')
    <a href="{{ route('pos.grid') }}" class="btn btn-sm btn-primary" data-bs-toggle="tooltip"title="{{ __('Grid View') }}">
        <i class="ti ti-layout-grid text-white"></i>
    </a>
</div>

@endsection


@section('content')
    <div id="printableArea">
        <div class="row mt-3">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-body table-border-style">
                        <div class="table-responsive">
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
