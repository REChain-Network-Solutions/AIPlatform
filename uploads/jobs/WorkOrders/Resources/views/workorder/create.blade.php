@extends('layouts.app')
@section('page-title')
    {{__('Work Order')}}
@endsection
@push('script-page')
    <script src="{{ asset('js/jquery-ui.min.js') }}"></script>
    <script src="{{ asset('js/jquery.repeater.min.js') }}"></script>
    <script>
        var estimationSelector = "body";
        if ($(estimationSelector + " .repeater").length) {
            var $rowDragAndDrop = $("body .repeater tbody").sortable({
                handle: '.sort-handler'
            });
            var $estimationRepeater = $(estimationSelector + ' .repeater').repeater({
                initEmpty: true,
                defaultValues: {
                    'status': 1
                },
                show: function () {
                    $(".hidesearch").each(function() {
                        var basic_select = new Choices(this, {
                            searchEnabled: false,
                            removeItemButton: true,
                        });
                    });
                    $(this).slideDown();
                },
                hide: function (deleteEstimation) {
                    if (confirm('Are you sure you want to delete this record?')) {
                        $(this).slideUp(deleteEstimation);
                        $(this).remove();
                    }
                },
                ready: function (setIndexes) {
                    $rowDragAndDrop.on('drop', setIndexes);
                },
                isFirstItemUndeletable: false
            });

        }
    </script>
    <script>
        $(document).on('change', '.service_part_id', function () {
            var currentElement = $(this).closest('tr');
            var service_part_id = $(this).val();
            var url = '{{ route("workorder.service.part") }}';
            $.ajax({
                url: url,
                headers: {
                    'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
                },
                data: {
                    id: service_part_id,
                },
                contentType: false,
                type: 'GET',
                success: function (data) {
                    currentElement.find('.quantity').val(1);
                    currentElement.find('.amount').val(data.price);
                    currentElement.find('.unit').val(data.unit);
                    currentElement.find('.description').val(data.description);
                },
            });
        });

    </script>
@endpush
@section('breadcrumb')
    <ul class="breadcrumb mb-0">
        <li class="breadcrumb-item">
            <a href="{{route('dashboard')}}">{{__('Dashboard')}}</a>
        </li>
        <li class="breadcrumb-item">
            <a href="{{route('workorder.index')}}">{{__('Work Order')}}</a>
        </li>
        <li class="breadcrumb-item active">
            <a href="#">{{__('Create')}}</a>
        </li>
    </ul>
@endsection

@section('content')
<div class="row">
    {{Form::open(array('url'=>'workorder','method'=>'post'))}}
        <div class="col-lg-12 col-md-12">
            <div class="card">
                <div class="card-body">
                    <div class="info-group row">
                        <div class="form-group col-md-6">
                            <div class="form-group">
                                {{Form::label('wo_id',__('Workorder Number'),array('class'=>'form-label'))}}
                                <span class="text-danger">*</span>
                                <div class="input-group">
                                        <span class="input-group-text ">
                                          {{workOrderPrefix()}}
                                        </span>
                                    {{Form::text('wo_id',$workOrderNumber,array('class'=>'form-control','placeholder'=>__('Enter Workorder Number')))}}
                                </div>
                            </div>
                        </div>
                        <div class="form-group col-md-6 col-lg-6">
                            {{Form::label('wo_detail',__('WO Detail'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
                            {{Form::textarea('wo_detail',old('wo_detail'),array('class'=>'form-control','rows'=>1,'required'=>'required'))}}
                        </div>
                        <div class="form-group col-md-6 col-lg-6">
                            {{ Form::label('type', __('Type'),['class'=>'form-label']) }} <span class="text-danger">*</span>
                            {!! Form::select('type', $woTypes, old('type'),array('class' => 'form-control hidesearch','required'=>'required')) !!}
                        </div>
                        <div class="form-group col-md-6">
                            {{ Form::label('client', __('Client'),['class'=>'form-label']) }} <span class="text-danger">*</span>
                            {!! Form::select('client', $clients, old('client'),array('class' => 'form-control hidesearch','required'=>'required')) !!}
                        </div>
                        <div class="form-group col-md-6">
                            {{ Form::label('asset', __('Asset'),['class'=>'form-label']) }} <span class="text-danger">*</span>
                            {!! Form::select('asset', $assets, old('asset'),array('class' => 'form-control hidesearch','required'=>'required')) !!}
                        </div>
                        <div class="form-group col-md-6">
                            {{Form::label('due_date',__('Due Date'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
                            {{Form::date('due_date',old('due_date'),array('class'=>'form-control','required'=>'required'))}}
                        </div>
                        <div class="form-group col-md-6">
                            {{ Form::label('priority', __('Priority'),['class'=>'form-label']) }} <span class="text-danger">*</span>
                            {!! Form::select('priority', $priority, old('priority'),array('class' => 'form-control hidesearch','required'=>'required')) !!}
                        </div>

                        <div class="form-group col-md-6">
                            {{ Form::label('assign', __('Assign'),['class'=>'form-label']) }} <span class="text-danger">*</span>
                            {!! Form::select('assign', $users, old('assign'),array('class' => 'form-control hidesearch')) !!}
                        </div>
                        <div class="form-group col-md-6">
                            {{Form::label('notes',__('Notes'),array('class'=>'form-label')) }}
                            {{Form::textarea('notes',old('notes'),array('class'=>'form-control','rows'=>1))}}
                        </div>
                        <hr>
                        <div class="form-group col-md-6">
                            {{Form::label('preferred_date',__('Preferred Date'),array('class'=>'form-label')) }}
                            {{Form::date('preferred_date',old('preferred_date'),array('class'=>'form-control'))}}
                        </div>
                        <div class="form-group col-md-6">
                            {{ Form::label('preferred_time', __('Preferred Time'),['class'=>'form-label']) }}
                            {!! Form::select('preferred_time', $time, old('preferred_time'),array('class' => 'form-control hidesearch')) !!}
                        </div>
                        <div class="form-group  col-md-6">
                            {{Form::label('preferred_note',__('Preference Note'),array('class'=>'form-label')) }}
                            {{Form::textarea('preferred_note',old('preferred_note'),array('class'=>'form-control','rows'=>1))}}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="">
            <div class="col-lg-12 col-md-12">
                <div class="card repeater services">
                    <div class="card-header">
                        <div class="row align-items-center g-2">
                            <div class="col">
                                <h5>{{ __('Services') }}</h5>
                            </div>
                            <div class="col-auto">
                                <a class="btn btn-secondary btn-sm" href="#" data-repeater-create=""> <i
                                        class="ti ti-circle-plus align-text-bottom"></i>{{ __('Add Service') }}</a>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <table class="display dataTable cell-border" data-repeater-list="services">
                            <thead>
                            <tr>
                                <th>{{__('Service')}}</th>
                                <th>{{__('Quantity')}}</th>
                                <th>{{__('Unit')}}</th>
                                <th>{{__('Amount')}}</th>
                                <th>{{__('Description')}}</th>
                                <th>#</th>
                            </tr>
                            </thead>
                            <tbody data-repeater-item>
                            <tr>
                                <td width="30%">
                                    {{Form::select('service_part_id',$services,null,array('class'=>'form-control hidesearch service_part_id'))}}
                                </td>
                                <td>
                                    {{Form::number('quantity',null,array('class'=>'form-control quantity'))}}
                                </td>
                                <td>
                                    {!! Form::text('unit', null, array('class'=>'form-control unit', 'readonly')) !!}
                                    {{-- <div class="input-group unit"></div> --}}
                                </td>
                                <td>
                                    {{Form::number('amount',null,array('class'=>'form-control amount'))}}
                                </td>
                                <td>
                                    {{Form::textarea('description',null,array('class'=>'form-control description','rows'=>1))}}
                                </td>
                                <td>
                                    <a class="text-danger" data-repeater-delete href="#"> <i data-feather="trash-2"></i></a>
                                </td>
                            </tr>
                            </tbody>

                        </table>
                    </div>
                </div>
                <div class="card repeater parts">
                    <div class="card-header">
                        <div class="row align-items-center g-2">
                            <div class="col">
                                <h5>{{ __('Parts') }}</h5>
                            </div>
                            <div class="col-auto">
                                <a class="btn btn-secondary btn-sm" href="#" data-repeater-create=""> <i
                                        class="ti ti-circle-plus align-text-bottom"></i>{{ __('Add Part') }}</a>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <table class="display dataTable cell-border" data-repeater-list="parts">
                            <thead>
                            <tr>
                                <th>{{__('Part')}}</th>
                                <th>{{__('Quantity')}}</th>
                                <th>{{__('Unit')}}</th>
                                <th>{{__('Amount')}}</th>
                                <th>{{__('Description')}}</th>
                                <th>#</th>
                            </tr>
                            </thead>
                            <tbody data-repeater-item>
                            <tr>
                                <td width="30%">
                                    {{Form::select('service_part_id',$parts,null,array('class'=>'form-control hidesearch service_part_id'))}}
                                </td>
                                <td>
                                    {{Form::number('quantity',null,array('class'=>'form-control quantity'))}}
                                </td>
                                <td>
                                    {{-- <div class="input-group unit"> </div> --}}
                                    {!! Form::text('unit', null, array('class'=>'form-control unit', 'readonly')) !!}
                                </td>
                                <td>
                                    {{Form::number('amount',null,array('class'=>'form-control amount'))}}
                                </td>
                                <td>
                                    {{Form::textarea('description',null,array('class'=>'form-control description','rows'=>1))}}
                                </td>
                                <td>
                                    <a class="text-danger" data-repeater-delete href="#"> <i data-feather="trash-2"></i></a>
                                </td>
                            </tr>
                            </tbody>

                        </table>
                    </div>
                </div>

            </div>
            <div class="row">
                <div class=" text-end mb-3  ">
                    {{Form::submit(__('Create'),array('class'=>'btn btn-secondary btn-rounded'))}}
                </div>
            </div>
        </div>

        {{Form::close()}}
    </div>
@endsection
