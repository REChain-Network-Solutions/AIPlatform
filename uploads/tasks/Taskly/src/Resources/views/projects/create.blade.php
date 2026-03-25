{{ Form::open(array('route' => 'projects.store','enctype'=>'multipart/form-data','class'=>'needs-validation','novalidate')) }}
<div class="modal-body">
    <div class="text-end">
        @if (module_is_active('AIAssistant'))
            @include('aiassistant::ai.generate_ai_btn',['template_module' => 'project','module'=>'Taskly'])
        @endif
    </div>
    <div class="row">
        <div class="form-group col-md-12">
            {{ Form::label('projectname', __('Name'),['class'=>'form-label']) }}<x-required></x-required>
            {{ Form::text('name', '', array('class' => 'form-control','required'=>'required','id'=>"projectname",'placeholder'=> __('Project Name'))) }}
        </div>

        <div class="form-group col-md-12">
            {{ Form::label('users_list', __('Users'),['class'=>'form-label']) }}
            <select class=" multi-select choices" id="users_list" name="users_list[]"  multiple="multiple" data-placeholder="{{ __('Select Users ...') }}">
                @foreach($workspace_users as $user)
                        <option value="{{$user->email}}">{{$user->name}} - {{$user->email}}</option>
                @endforeach
            </select>
            <p class="text-danger d-none" id="user_validation">{{__('Users filed is required.')}}</p>
        </div>
        <div class="form-group col-md-12">
            {{ Form::label('projectname', __('Description'),['class'=>'form-label']) }}<x-required></x-required>
            {{ Form::textarea('description', '', array('class' => 'form-control','rows'=>3,'required'=>'required','id'=>"description",'placeholder'=> __('Add Description'))) }}
        </div>
        @if(module_is_active('CustomField') && !$customFields->isEmpty())
            <div class="col-md-12">
                <div class="tab-pane fade show" id="tab-2" role="tabpanel">
                    @include('custom-field::formBuilder')
                </div>
            </div>
        @endif
    </div>
</div>
<div class="modal-footer">
    <input type="button" value="{{__('Cancel')}}" class="btn  btn-light" data-bs-dismiss="modal">
    <input type="submit" value="{{__('Create')}}" class="btn  btn-primary" id="submit">
</div>
{{ Form::close() }}

<script>
    $(function(){
        $("#submit").click(function() {
            var user =  $("#users_list option:selected").length;
            if(user == 0){
            $('#user_validation').removeClass('d-none')
                return false;
            }else{
            $('#user_validation').addClass('d-none')
            }
        });
    });
</script>
