@extends('layouts.main')
@section('page-title')
    {{__('Manage POS Order')}}
@endsection
@section('page-breadcrumb')
   {{__('POS Order')}}
@endsection
@push('css')
    <link rel="stylesheet" href="{{ asset('css/datatable/buttons.dataTables.min.css') }}">
@endpush

@push('scripts')
    <script src="{{ asset('js/letter.avatar.js') }}"></script>
@endpush

@section('page-action')
    <div class="d-flex">
        @stack('addButtonHook')
        <a href="{{ route('pos.report') }}" class="btn btn-sm btn-primary btn-icon" data-bs-toggle="tooltip"
        title="{{ __('List View') }}">
            <i class="ti ti-list text-white"></i>
        </a>
    </div>
@endsection

@section('content')
<div class="row row-gap-2 mb-4">
    @if(count($posPayments) > 0)
        @foreach ($posPayments as $posPayment)
            <div class="col-xxl-3 col-xl-4 col-md-6">
                <div class="card user-card">
                    <div class="card-header border border-bottom p-3 h-100">
                        <div class="user-img-wrp d-flex align-items-center">
                            <div class="user-image rounded border-2 border border-primary">
                                <img
                                    @if(!empty($posPayment->avatar))
                                src="{{(!empty($posPayment->avatar))? get_file("profile/".$posPayment->avatar): asset(url("./assets/img/clients/160x160/img-1.png"))}}"
                                @else
                                @if($posPayment->customer_id == 0)
                                avatar="{{__('Walk-in Customer')}}"
                                @else
                                avatar="@if(module_is_active('Account')) {{!empty($posPayment->customer) ? $posPayment->customer->name : $posPayment->user->name}} @else ''
                                @endif "
                                @endif
                                @endif
                                alt="user-image" class="h-100 w-100">
                            </div>
                            <div class="user-content">      
                                @if($posPayment->customer_id == 0)
                                    <a href="{{  route('pos.show',\Crypt::encrypt($posPayment->id)) }}" data-title="{{__('Purchase Details')}}" class="action-item text-primary mt-2">
                                        <h4 class="mb-2">{{ ucfirst('Walk-in Customer') }}</h4>
                                    </a>
                                @else
                                    @if(module_is_active('Account'))
                                        <a href="{{  route('pos.show',\Crypt::encrypt($posPayment->id)) }}" data-title="{{__('Purchase Details')}}" class="action-item text-primary mt-2">
                                            <h4 class="mb-2">{{ ucfirst(!empty($posPayment->customer) ? $posPayment->customer->name :$posPayment->user->name) }}</h4>
                                        </a>
                                    @else
                                        <a href="{{  route('pos.show',\Crypt::encrypt($posPayment->id)) }}" data-title="{{__('Purchase Details')}}" class="action-item text-primary mt-2">
                                            <h4 class="mb-2">{{ ucfirst(!empty($posPayment->user) ? $posPayment->user->name : '-') }}</h4>
                                        </a>
                                    @endif
                                @endif
                                <span class="text-dark text-md">{{ !empty($posPayment->warehouse_id) ? $posPayment->warehouse->name : '-' }}</span>
                            </div>
                        </div>
                    </div>
                    <div class="card-body text-center p-3">
                        <div class="bottom-icons d-flex flex-wrap align-items-center justify-content-between">
                            <div class="edit-btn-wrp d-flex flex-wrap align-items-center">
                                @permission('pos show')
                                <a href="{{ route('pos.show',\Crypt::encrypt($posPayment->id)) }}"
                                    data-bs-whatever="{{ __('Common case Details') }}"
                                    data-title="{{ __('View') }}"
                                    data-bs-original-title="{{ __('View') }}" data-bs-toggle="tooltip"
                                    data-bs-placement="top" class="btn btn-sm border">
                                    <svg width="16" height="16" viewBox="0 0 15 15" fill="none"
                                        xmlns="http://www.w3.org/2000/svg">
                                        <path
                                            d="M7.9997 10.8869C6.40637 10.8869 5.11304 9.59353 5.11304 8.00019C5.11304 6.40686 6.40637 5.11353 7.9997 5.11353C9.59304 5.11353 10.8864 6.40686 10.8864 8.00019C10.8864 9.59353 9.59304 10.8869 7.9997 10.8869ZM7.9997 6.11353C6.9597 6.11353 6.11304 6.96019 6.11304 8.00019C6.11304 9.04019 6.9597 9.88686 7.9997 9.88686C9.0397 9.88686 9.88637 9.04019 9.88637 8.00019C9.88637 6.96019 9.0397 6.11353 7.9997 6.11353Z"
                                            fill="#060606" />
                                        <path
                                            d="M7.99967 14.0134C5.493 14.0134 3.12633 12.5467 1.49967 10C0.792999 8.90003 0.792999 7.10669 1.49967 6.00003C3.133 3.45336 5.49967 1.98669 7.99967 1.98669C10.4997 1.98669 12.8663 3.45336 14.493 6.00003C15.1997 7.10003 15.1997 8.89336 14.493 10C12.8663 12.5467 10.4997 14.0134 7.99967 14.0134ZM7.99967 2.98669C5.84633 2.98669 3.78633 4.28003 2.34633 6.54003C1.84633 7.32003 1.84633 8.68003 2.34633 9.46003C3.78633 11.72 5.84633 13.0134 7.99967 13.0134C10.153 13.0134 12.213 11.72 13.653 9.46003C14.153 8.68003 14.153 7.32003 13.653 6.54003C12.213 4.28003 10.153 2.98669 7.99967 2.98669Z"
                                            fill="#060606" />
                                    </svg>
                                </a>
                                @endpermission
                            </div>
                            @permission('pos show')
                                <a class="badge p-2 px-3 bg-primary" href="{{ route('pos.show',\Crypt::encrypt($posPayment->id)) }}">
                                    {{ \workdo\Pos\Entities\Pos::posNumberFormat($posPayment->pos_id) }}
                                </a>
                            @else
                                <a class="badge p-2 px-3 bg-primary" href="#">
                                    {{ \workdo\Pos\Entities\Pos::posNumberFormat($posPayment->pos_id) }}
                                </a>
                            @endpermission
                        </div>
                    </div>
                </div>
            </div>
        @endforeach
        {!! $posPayments->links('vendor.pagination.global-pagination') !!}
    @else
        <div class="row product-row justify-content-center">
            <div class="text-center">
                <i class="fas fa-folder-open text-gray" style="font-size: 48px;"></i>
                <h2>{{ __('Opps...') }}</h2>
                <h6> {!! __('No data Found.') !!} </h6>
            </div>
        </div>
    @endif
</div>
@endsection
