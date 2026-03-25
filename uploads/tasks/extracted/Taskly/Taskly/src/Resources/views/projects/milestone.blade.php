@if($project && $currentWorkspace)
    {{ Form::open(array('route' => array('projects.milestone.store', $project->id),'class'=>'needs-validation','novalidate')) }}
        <div class="modal-body">
            <div class="text-end">
                @if (module_is_active('AIAssistant'))
                    @include('aiassistant::ai.generate_ai_btn',['template_module' => 'milestone','module'=>'Taskly'])
                @endif
            </div>
            <div class="row">
                <div class="col-md-8">
                    <div class="form-group">
                        {{ Form::label('milestone-title', __('Milestone Title'),['class'=>'form-label']) }}<x-required></x-required>
                        {{ Form::text('title', null, array('class' => 'form-control','required'=>'required','id'=>"milestone-title",'placeholder'=> __('Enter Title'))) }}
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="form-group">
                        {{ Form::label('milestone-status', __('Status'),['class'=>'form-label']) }}<x-required></x-required>
                        {{ Form::select('status', ['incomplete'=>__('Incomplete'),'complete'=>__('Complete')],null, array('class' => 'form-control','id'=>'status','required'=>'required')) }}
                    </div>
                </div>
            </div>

            <div class="form-group">
                {{ Form::label('budget', __('Milestone Cost'),['class'=>'form-label']) }}<x-required></x-required>
                <div class="input-group mb-3">
                    <span class="input-group-text">{{ (!empty($currentWorkspace->currency)) ? $currentWorkspace->currency : '$'}}</span>
                    {{ Form::number('cost', null, array('class' => 'form-control currency_input','required'=>'required','id'=>"budget",'placeholder'=> __('Enter Cost'),"min"=>0)) }}
                </div>
            </div>
            <div class="form-group">
                {{ Form::label('task-summary', __('Summary'),['class'=>'form-label']) }}<x-required></x-required>
                {{ Form::textarea('summary', null, array('class' => 'form-control','required'=>'required','rows'=>3,'id'=>"milestone-title",'placeholder'=> __('Enter Summary'))) }}
            </div>
        </div>
        <div class="modal-footer">
            <button type="button" class="btn  btn-light" data-bs-dismiss="modal">{{ __('Cancel')}}</button>
            <input type="submit" value="{{ __('Create')}}" class="btn  btn-primary">
        </div>
    {{ Form::close() }}
@else
    <div class="container mt-5">
        <div class="card">
            <div class="card-body p-4">
                <div class="page-error">
                    <div class="page-inner">
                        <h1>404</h1>
                        <div class="page-description">
                            {{ __('Page Not Found') }}
                        </div>
                        <div class="page-search">
                            <p class="text-muted mt-3">{{ __("It's looking like you may have taken a wrong turn. Don't worry... it happens to the best of us. Here's a little tip that might help you get back on track.")}}</p>
                            <div class="mt-3">
                                <a class="btn-return-home badge-blue" href="{{route('home')}}"><i class="fas fa-reply"></i> {{ __('Return Home')}}</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
@endif
