{{ Form::model($servicePart, array('route' => array('services-parts.update', $servicePart->id), 'method' => 'PUT')) }}
<div class="modal-body wrapper">
    <div class="row">
        <div class="form-group col-md-6">
            {{Form::label('title',__('Title'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::text('title',null,array('class'=>'form-control','placeholder'=>__('Enter title'),'required'=>'required'))}}
        </div>
        <div class="form-group col-md-6">
            {{Form::label('sku',__('SKU'),array('class'=>'form-label'))}} <span class="text-danger">*</span>
            {{Form::text('sku',null,array('class'=>'form-control','placeholder'=>__('Enter sku'),'required'=>'required'))}}
        </div>
        <div class="form-group col-md-6">
            {{Form::label('price',__('Price'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::text('price',null,array('class'=>'form-control','placeholder'=>__('Enter price'),'required'=>'required'))}}
        </div>
        <div class="form-group col-md-6">
            {{Form::label('unit',__('Unit'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::text('unit',null,array('class'=>'form-control','placeholder'=>__('Enter unit')))}}
        </div>
        <div class="form-group col-md-12">
            {{Form::label('description',__('Description'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::text('description',null,array('class'=>'form-control','placeholder'=>__('Enter description')))}}
        </div>
    </div>
    @if($servicePart->type=='service')
        <div class="service_tasks">
            <div class="row">
                <div class="col-sm-12">
                    <a href="#" class="btn btn-seconadry btn-xs service_task_clone float-end"><i
                                class="ti ti-plus"></i></a>
                </div>
            </div>
            @foreach($servicePart->serviceTasks as $task)
                <input type="hidden" name="id[]" value="{{$task->id}}">
                <div class="row service_task">
                    <div class="form-group col-md-4">
                        {{Form::label('task',__('Service Task Title'),array('class'=>'form-label')) }}
                        {{Form::text('task[]',$task->task,array('class'=>'form-control','placeholder'=>__('service task title')))}}
                    </div>
                    <div class="form-group col-md-3">
                        {{Form::label('duration',__('Duration'),array('class'=>'form-label')) }}
                        {{Form::text('duration[]',$task->duration,array('class'=>'form-control','placeholder'=>__('like 1 Hour 20 Min')))}}
                    </div>
                    <div class="form-group col-md-4">
                        {{Form::label('task_description',__('Description'),array('class'=>'form-label')) }} <span
                                class="text-danger">*</span>
                        {{Form::text('task_description[]',$task->description,array('class'=>'form-control','placeholder'=>__('description')))}}
                    </div>
                    <div class="col-auto">
                        <a href="#" class="f-20 text-danger service_task_remove"  data-val="{{$task->id}}"> <i
                                    class="ti ti-trash"></i></a>
                    </div>
                </div>
            @endforeach
            @if(count($servicePart->serviceTasks)==0)
                <div class="row service_task">
                    <div class="form-group col-md-4">
                        {{Form::label('task',__('Service Task Title'),array('class'=>'form-label')) }}
                        {{Form::text('task[]',null,array('class'=>'form-control','placeholder'=>__('service task title')))}}
                    </div>
                    <div class="form-group col-md-3">
                        {{Form::label('duration',__('Duration'),array('class'=>'form-label')) }}
                        {{Form::text('duration[]',null,array('class'=>'form-control','placeholder'=>__('like 1 Hour 20 Min')))}}
                    </div>
                    <div class="form-group col-md-4">
                        {{Form::label('task_description',__('Description'),array('class'=>'form-label')) }} <span
                                class="text-danger">*</span>
                        {{Form::text('task_description[]',null,array('class'=>'form-control','placeholder'=>__('description')))}}
                    </div>
                    <div class="col-auto">
                        <a href="#" class="f-20 text-danger service_task_remove"> <i
                                    class="ti ti-trash"></i></a>
                    </div>
                </div>
            @endif
            <div class="service_task_results"></div>
        </div>
    @endif
</div>
<div class="modal-footer">
    {{Form::submit(__('Update'),array('class'=>'btn btn-secondary ml-10'))}}
</div>
{{Form::close()}}
<script>
    $('.wrapper').on('click', '.service_task_remove', function () {
        var id=$(this).data('val');
        if(id!=''){
            if (confirm('Are you sure you want to delete this element?')) {
                $.ajax({
                    url: '{{route('service.task.destroy')}}',
                    type: 'DELETE',
                    headers: {
                        'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
                    },
                    data: {
                        'id': id
                    },
                    cache: false,
                    success: function (data) {

                    },
                });
            }
        }
        $('.schedule_remove').closest('.wrapper').find('.schedule').not(':first').last().remove();
    });

    $('.wrapper').on('click', '.service_task_remove', function () {
        $('.service_task_remove').closest('.wrapper').find('.service_task').not(':first').last().remove();
    });
    $('.wrapper').on('click', '.service_task_clone', function () {
        $('.service_task_clone').closest('.wrapper').find('.service_task').first().clone().appendTo('.service_task_results');
    });
</script>


