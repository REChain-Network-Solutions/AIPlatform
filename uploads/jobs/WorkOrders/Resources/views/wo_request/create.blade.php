{{Form::open(array('url'=>'wo-request','method'=>'post'))}}
<div class="modal-body">
    <div class="row">
        <div class="form-group col-md-12">
            {{Form::label('request_detail',__('Request Detail'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::textarea('request_detail',null,array('class'=>'form-control','rows'=>1,'required'=>'required'))}}
        </div>
        <div class="form-group col-md-6">
            {{ Form::label('client', __('Client'),['class'=>'form-label']) }} <span class="text-danger">*</span>
            {!! Form::select('client', $clients, null,array('class' => 'form-control hidesearch','required'=>'required')) !!}
        </div>
        <div class="form-group col-md-6">
            {{ Form::label('asset', __('Asset'),['class'=>'form-label']) }} <span class="text-danger">*</span>
            {!! Form::select('asset', $assets, null,array('class' => 'form-control hidesearch','required'=>'required')) !!}
        </div>
        <div class="form-group col-md-6">
            {{Form::label('due_date',__('Due Date'),array('class'=>'form-label')) }} <span class="text-danger">*</span>
            {{Form::date('due_date',null,array('class'=>'form-control','required'=>'required'))}}
        </div>
        <div class="form-group col-md-6">
            {{ Form::label('priority', __('Priority'),['class'=>'form-label']) }} <span class="text-danger">*</span>
            {!! Form::select('priority', $priority, null,array('class' => 'form-control hidesearch','required'=>'required')) !!}
        </div>
        <div class="form-group col-md-6">
            {{ Form::label('status', __('Status'),['class'=>'form-label']) }} <span class="text-danger">*</span>
            {!! Form::select('status', $status, null,array('class' => 'form-control hidesearch','required'=>'required')) !!}
        </div>
        <div class="form-group col-md-6">
            {{ Form::label('assign', __('Assign'),['class'=>'form-label']) }} <span class="text-danger">*</span>
            {!! Form::select('assign', $users, null,array('class' => 'form-control hidesearch')) !!}
        </div>
        <div class="form-group col-md-12">
            {{Form::label('notes',__('Notes'),array('class'=>'form-label')) }}
            {{Form::textarea('notes',null,array('class'=>'form-control','rows'=>2))}}
        </div>
        <hr>
        <div class="form-group col-md-6">
            {{Form::label('preferred_date',__('Preferred Date'),array('class'=>'form-label')) }}
            {{Form::date('preferred_date',null,array('class'=>'form-control'))}}
        </div>
        <div class="form-group col-md-6">
            {{ Form::label('preferred_time', __('Preferred Time'),['class'=>'form-label']) }}
            {!! Form::select('preferred_time', $time, null,array('class' => 'form-control hidesearch')) !!}
        </div>
        <div class="form-group col-md-12">
            {{Form::label('preferred_note',__('Preference Note'),array('class'=>'form-label')) }}
            {{Form::textarea('preferred_note',null,array('class'=>'form-control','rows'=>1))}}
        </div>

    </div>
</div>
<div class="modal-footer">
    {{Form::submit(__('Create'),array('class'=>'btn btn-secondary ml-10'))}}
</div>
{{Form::close()}}

