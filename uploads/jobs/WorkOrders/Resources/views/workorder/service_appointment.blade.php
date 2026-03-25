{{ Form::model($serviceAppointment, array('route' => array('workorder.service.appointment.store', [$wo_id]), 'method' => 'PUT')) }}
<div class="modal-body">
    <div class="row">
        <div class="form-group col-lg-6 col-md-6">
            {{Form::label('start_date',__('Start Date'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::date('start_date',null,array('class'=>'form-control','required'))}}
        </div>
        <div class="form-group col-lg-6 col-md-6">
            {{Form::label('start_time',__('Start Time'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::time('start_time',null,array('class'=>'form-control','required'))}}
        </div>
        <div class="form-group col-lg-6 col-md-6">
            {{Form::label('end_date',__('End Date'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::date('end_date',null,array('class'=>'form-control','required'))}}
        </div>
        <div class="form-group col-lg-6 col-md-6">
            {{Form::label('end_time',__('End Time'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::time('end_time',null,array('class'=>'form-control','required'))}}
        </div>
        <div class="form-group">
            {{ Form::label('status', __('Status'),['class'=>'form-label']) }}
            {!! Form::select('status', $status, null,array('class' => 'form-control hidesearch')) !!}
        </div>
        <div class="form-group ">
            {{Form::label('notes',__('Notes'),array('class'=>'form-label')) }}
            {{Form::textarea('notes',null,array('class'=>'form-control','placeholder'=>__('Notes'),'rows'=>2))}}
        </div>
    </div>
</div>
<div class="modal-footer">
    {{Form::submit(__('Save'),array('class'=>'btn btn-secondary ml-10'))}}
</div>
{{Form::close()}}
