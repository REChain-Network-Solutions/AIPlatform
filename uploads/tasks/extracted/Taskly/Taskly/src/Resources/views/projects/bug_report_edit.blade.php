@if($project && $currentWorkspace && $bug)

    {{ Form::model($bug, array('route' => array('projects.bug.report.update',[$project->id,$bug->id]), 'method' => 'Post','class'=>'needs-validation','novalidate')) }}
    @csrf
    <div class="modal-body">
        <div class="text-end">
            @if (module_is_active('AIAssistant'))
                @include('aiassistant::ai.generate_ai_btn',['template_module' => 'project bug','module'=>'Taskly'])
            @endif
        </div>
        <div class="row">
            <div class="form-group col-md-6">
                <label class="form-label">{{ __('Title')}}</label><x-required></x-required>
                <input type="text" class="form-control" id="task-title" placeholder="{{ __('Enter Title')}}" name="title" value="{{$bug->title}}" required>
            </div>

            <div class="form-group col-md-6">
                <label for="task-priority" class="form-label">{{ __('Priority')}}</label><x-required></x-required>
                <select class="form-control" name="priority" id="task-priority" required>
                    <option value="Low" @if($bug->priority=='Low') selected @endif>{{ __('Low')}}</option>
                    <option value="Medium" @if($bug->priority=='Medium') selected @endif>{{ __('Medium')}}</option>
                    <option value="High" @if($bug->priority=='High') selected @endif>{{ __('High')}}</option>
                </select>
            </div>
            <div class="form-group col-md-6">
                <label for="assign_to" class="form-label">{{ __('Assign To')}}</label><x-required></x-required>
                <select class="form-control" id="assign_to" name="assign_to" required>
                    @foreach($users as $u)
                        <option @if($bug->assign_to==$u->id) selected @endif value="{{$u->id}}">{{$u->name}} - {{$u->email}}</option>
                    @endforeach
                </select>
            </div>
            <div class="form-group col-md-6">
                <label for="status" class="form-label">{{ __('Status')}}</label><x-required></x-required>
                <select class="form-control" id="status" name="status" required>
                    @foreach($arrStatus as $id => $status)
                        <option @if($bug->status==$id) selected @endif value="{{$id}}">{{__($status)}}</option>
                    @endforeach
                </select>
            </div>
            <div class="form-group col-md-12 mb-0">
                <label for="task-description" class="form-label">{{ __('Description')}}</label>
                <textarea class="form-control" id="task-description" rows="3" name="description" placeholder ="Enter Description">{{$bug->description}}</textarea>
            </div>


            @if(module_is_active('CustomField') && !$customFields->isEmpty())
                <div class="form-group col-md-12">
                    <div class="tab-pane fade show col-form-label" id="tab-2" role="tabpanel">
                        @include('custom-field::formBuilder')
                    </div>
                </div>
            @endif
        </div>
    </div>
    <div class="modal-footer">
        <button type="button" class="btn  btn-light" data-bs-dismiss="modal">{{ __('Cancel')}}</button>
        <input type="submit" value="{{ __('Update')}}" class="btn  btn-primary">
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
