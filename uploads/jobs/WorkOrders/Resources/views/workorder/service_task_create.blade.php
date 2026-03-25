{{Form::open(array('route'=>array('workorder.service.task.store',$workorder->id),'method'=>'post'))}}
<div class="modal-body">
    <div class="row">
        <div class="form-group">
            {{ Form::label('service', __('Service'),['class'=>'form-label']) }} <span class="text-danger">*</span>
            <select class="form-control hidesearch" id="service" name="service" required="">
                <option value="">{{__('Select Service')}}</option>
                @foreach($woServices as $service)
                    @if(!empty($service->serviceParts))
                        <option value="{{$service->serviceParts->id}}">{{$service->serviceParts->title}}</option>
                    @endif
                @endforeach
            </select>
        </div>
        <div class="form-group">
            {{Form::label('service_task',__('Task'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::text('service_task',null,array('class'=>'form-control','placeholder'=>__('service task title'),'required'))}}
        </div>
        <div class="form-group">
            {{Form::label('duration',__('Duration'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::text('duration',null,array('class'=>'form-control','placeholder'=>__('like 1 Hour 20 Min'),'required'))}}
        </div>
        <div class="form-group ">
            {{Form::label('description',__('Description'),array('class'=>'form-label')) }}
            {{Form::text('description',null,array('class'=>'form-control','placeholder'=>__('description')))}}
        </div>
        <div class="form-group">
            {{ Form::label('status', __('Status'),['class'=>'form-label']) }}
            {!! Form::select('status', $status, null,array('class' => 'form-control hidesearch')) !!}
        </div>
    </div>
</div>
<div class="modal-footer">
    {{Form::submit(__('Create'),array('class'=>'btn btn-secondary ml-10'))}}
</div>
{{Form::close()}}
