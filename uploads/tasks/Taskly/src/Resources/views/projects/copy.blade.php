{{ Form::model($project, array('route' => array('project.copy.store', $project->id), 'method' => 'POST')) }}
<div class="modal-body">
    <div class="row">
        <div class="col-12">
            <div class="form-group project p-3 mb-3 rounded">
                <div class="form-check">
                    {{ Form::checkbox('project','all','', ['class' => 'form-check-input ','id'=>'all']) }}
                    {{ Form::label('all', __('All'),['class'=>'form-check-label'])}}
                </div>
            </div>
        </div>
        <div class="col-12">
            <div class="form-group project p-3 rounded">
            {{-- project task --}}
            <div class="mb-3">
                <div class="form-check">
                    {{ Form::checkbox('task[]','task','', ['class' => 'form-check-input checkbox','id'=>'task']) }}
                    {{ Form::label('task', __('Task'),['class'=>'form-check-label'])}}
                </div>
            </div>
            <div class="row mx-4">
                <div class="col-4">
                    <div class="form-check">
                        {{ Form::checkbox('task[]','sub_task','', ['class' => 'form-check-input checkbox task','id'=>'sub_task']) }}
                        {{ Form::label('sub_task', __('Sub Task'),['class'=>'form-check-label'])}}
                    </div>
                </div>
                <div class="col-4">
                    <div class="form-check">
                        {{ Form::checkbox('task[]','task_comment','', ['class' => 'form-check-input checkbox task','id'=>'task_comment']) }}
                        {{ Form::label('task_comment', __('Comment'),['class'=>'form-check-label'])}}
                    </div>
                </div>
                <div class="col-4">
                    <div class="form-check">
                        {{ Form::checkbox('task[]','task_files','', ['class' => 'form-check-input checkbox task','id'=>'task_files']) }}
                        {{ Form::label('task_files', __('Files'),['class'=>'form-check-label'])}}
                    </div>
                </div>
            </div>
        </div>
        </div>
        <div class="col-12">
            <div class="form-group project p-3 rounded">
            {{-- project bug --}}
            <div class="mb-3">
                <div class="form-check">
                    {{ Form::checkbox('bug[]','bug','', ['class' => 'form-check-input checkbox','id'=>"bug"]) }}
                    {{ Form::label('bug', __('Bug'),['class'=>'form-check-label'])}}
                </div>
            </div>
            <div class="row mx-4">
                    <div class="col-4">
                        <div class="form-check">
                            {{ Form::checkbox('bug[]','bug_comment','', ['class' => 'form-check-input checkbox bug','id'=>'bug_comment']) }}
                            {{ Form::label('bug_comment', __('Comment'),['class'=>'form-check-label'])}}
                        </div>
                    </div>
                <div class="col-4">
                    <div class="form-check">
                        {{ Form::checkbox('bug[]','bug_files','', ['class' => 'form-check-input checkbox bug','id'=>'bug_files']) }}
                        {{ Form::label('bug_files', __('Files'),['class'=>'form-check-label'])}}
                    </div>
                </div>
            </div>
        </div>
        </div>
        <div class="col-6">
            <div class="form-group project p-3 mb-3 rounded">
                <div class="form-check">
                    {{ Form::checkbox('activity[]','activity','', ['class' => 'form-check-input checkbox','id'=>"activity"]) }}
                    {{ Form::label('activity', __('Activity'),['class'=>'form-check-label'])}}
                </div>
            </div>
        </div>
        <div class="col-6">
            <div class="form-group project p-3 mb-3 rounded">
                <div class="form-check">
                    {{ Form::checkbox('user[]','user','', ['class' => 'form-check-input checkbox','id'=>"user"]) }}
                    {{ Form::label('user', __('Team Member'),['class'=>'form-check-label'])}}
                </div>
            </div>
        </div>
        <div class="col-6">
            <div class="form-group project p-3 mb-3 rounded">
                <div class="form-check">
                    {{ Form::checkbox('client[]','client','', ['class' => 'form-check-input checkbox','id'=>"client"]) }}
                    {{ Form::label('client', __('Client'),['class'=>'form-check-label'])}}
                </div>
            </div>
        </div>
        <div class="col-6">
            <div class="form-group project p-3 mb-3 rounded">
                <div class="form-check">
                    {{ Form::checkbox('milestone[]','milestone','', ['class' => 'form-check-input checkbox','id'=>"milestone"]) }}
                    {{ Form::label('milestone', __('Milestone'),['class'=>'form-check-label'])}}
                </div>
            </div>
        </div>
        <div class="col-6">
            <div class="form-group project p-3 m-0 rounded">
                <div class="form-check">
                    {{ Form::checkbox('project_file[]','project_file','', ['class' => 'form-check-input checkbox','id'=>"project_file"]) }}
                    {{ Form::label('project_file', __('Project File'),['class'=>'form-check-label'])}}
                </div>
            </div>
        </div>

    </div>
</div>
<div class="modal-footer">
    <button type="button" class="btn  btn-light" data-bs-dismiss="modal">{{__('Cancel')}}</button>
    <button type="submit" class="btn  btn-primary">{{__('Copy')}}</button>
</div>

{{ Form::close() }}

<script>
    $(document).ready(function(){
        $('#all').on('click',function(){
            if(this.checked){
                $('.checkbox').each(function(){
                    this.checked = true;
                });
            }else{
                $('.checkbox').each(function(){
                    this.checked = false;
                });
            }
        });
    });
</script>

<script>
    $(document).ready(function(){
        $("#sub_task").click(function(){
            $("#task").prop("checked", true);
        });
        $("#task_comment").click(function(){
            $("#task").prop("checked", true);
        });
        $("#task_files").click(function(){
            $("#task").prop("checked", true);
        });
        $("#bug_comment").click(function(){
            $("#bug").prop("checked", true);
        });
        $("#bug_files").click(function(){
            $("#bug").prop("checked", true);
        });

        $('#task').on('click',function(){
            $('.task').each(function(){
                this.checked = false;
            });
        });
        $('#bug').on('click',function(){
            $('.bug').each(function(){
                this.checked = false;
            });
        });
    });
</script>

