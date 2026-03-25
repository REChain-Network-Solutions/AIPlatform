@extends('layouts.main')
@section('page-title')
    {{__('Manage POS Barcode Print')}}
@endsection
@section('page-breadcrumb')
    {{__('Product Barcode')}},
    {{__('POS Barcode Print')}}
@endsection

@push('scripts')
    <script type="text/javascript" src="{{ asset('js/html2pdf.bundle.min.js') }}"></script>
    <script>

        $(document).ready(function () {
            var b_id = $('#warehouse_id').val();
            getProduct(b_id);
        });
        $(document).on('change', 'select[name=warehouse_id]', function () {

            var warehouse_id = $(this).val();
            getProduct(warehouse_id);
        });

        function getProduct(bid) {

            $.ajax({
                url: '{{route('pos.getproduct')}}',
                type: 'POST',
                data: {
                    "warehouse_id": bid, "_token": "{{ csrf_token() }}",
                },

                success: function (data) {
                    $('#product_id').empty();

                    $("#product_div").html('');
                    $('#product_div').append('<label for="product_id" class="form-label">{{__('Product')}}</label><span class="text-danger">*</span>');
                    $('#product_div').append('<select class="form-label" id="product_id" name="product_id[]"  multiple></select>');
                    $('#product_id').append('<option value="">{{__('Select Product')}}</option>');

                    $.each(data, function (key, value) {
                        $('#product_id').append('<option value="' + key + '">' + value + '</option>');
                    });
                    var multipleCancelButton = new Choices('#product_id', {
                        removeItemButton: true,
                    });

                }

            });
        }


    </script>
@endpush


@section('page-action')
    <div class="float-end">
        <a href="{{ route('pos.barcode') }}" class="btn btn-sm btn-primary" data-bs-toggle="tooltip" title="{{__('Back')}}">
            <i class="ti ti-arrow-left text-white"></i>
        </a>
    </div>
@endsection


@section('content')
    <div class="row mt-3">
        <div class="col-12">
            <div class="card">
                <div class="card-body">
                    {{Form::open(array('route'=>'pos.receipt','method'=>'post','class'=>'needs-validation','novalidate'))}}
                        <div class="row" id="printableArea">
                            <div class="col-lg-4">
                                <div class="form-group">
                                    {{Form::label('warehouse_id',__('Warehouse'),['class'=>'form-label'])}}<x-required></x-required>
                                    {{ Form::select('warehouse_id', $warehouses,'', array('class' => 'form-control select','id'=>'warehouse_id','required'=>'required')) }}
                                </div>
                            </div>
                            <div class="col-lg-4">
                                <div class="form-group" id="product_div">
                                    {{Form::label('product_id',__('Product'),['class'=>'form-label'])}}<x-required></x-required>
                                    <select class="form-control select" name="product_id[]" id="product_id" required >
                                    </select>
                                </div>
                            </div>
                            <div class="col-lg-4">
                                <div class="form-group">
                                    {{ Form::label('quantity', __('Quantity'),['class'=>'form-label']) }}<x-required></x-required>
                                    {{ Form::number('quantity',null, array('class' => 'form-control','placeholder'=>__('Enter Quantity'),'required'=>'required')) }}
                                </div>
                            </div>
                        </div>
                        <div class="col-md-12 pt-4">
                            <button class="btn btn-primary btn-icon float-end" type="submit"><i class="ti ti-printer text-white"></i> {{__('Print')}}</button>
                        </div>
                    {{Form::close()}}
                </div>
            </div>
        </div>
    </div>
@endsection


