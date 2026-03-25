@if ($task)
    <div class="modal-body">
        <div class="p-2">
            <div class="form-control-label">{{ __('Description') }}:</div>

            <p class="text-muted mb-4">
                {{ $task->description }}
            </p>

            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="form-control-label">{{ __('Create Date') }}</div>
                    <p class="mt-1">
                        {{ company_date_formate($task->created_at, $project->created_by, $project->workspace) }}</p>
                </div>
                <div class="col-md-3">
                    <div class="form-control-label">{{ __('Due Date') }}</div>
                    <p class="mt-1">
                        {{ company_date_formate($task->due_date, $project->created_by, $project->workspace) }}</p>
                </div>
                <div class="col-md-3">
                    <div class="form-control-label">{{ __('Assigned') }}</div>
                    @if ($users = $task->users())
                        @foreach ($users as $user)
                            <img alt="image" data-bs-toggle="tooltip" data-bs-placement="top"
                                title="{{ $user->name }}"
                                @if ($user->avatar) src="{{ get_file($user->avatar) }}" @else src="{{ get_file('avatar.png') }}" @endif
                                class="rounded-circle " width="20px" height="20px">
                        @endforeach
                    @endif
                </div>
                <div class="col-md-3">
                    <div class="form-control-label">{{ __('Milestone') }}</div>
                    @php($milestone = $task->milestone())
                    <p class="mt-1">
                        @if ($milestone)
                            {{ $milestone->title }}
                        @endif
                    </p>
                </div>
            </div>
        </div>

        <ul class="nav nav-pills mb-3" id="pills-tab" role="tablist" style="border: solid #eee;border-radius: 13px;">
            <li class="nav-item">
                <a class=" nav-link active" id="comments-tab" data-bs-toggle="tab" href="#comments-data" role="tab"
                    aria-controls="home" aria-selected="false"> {{ __('Comments') }} </a>
            </li>
            <li class="nav-item">
                <a class=" nav-link" id="file-tab" data-bs-toggle="tab" href="#file-data" role="tab"
                    aria-controls="profile" aria-selected="false"> {{ __('Files') }} </a>
            </li>
            <li class="nav-item">
                <a class=" nav-link" id="sub-task-tab" data-bs-toggle="tab" href="#sub-task-data" role="tab"
                    aria-controls="contact" aria-selected="true"> {{ __('Sub Task') }} </a>
            </li>
        </ul>
        <div class="tab-content" id="myTabContent">
            <div class="tab-pane fade active show" id="comments-data" role="tabpanel" aria-labelledby="home-tab">
                <ul class="list-unstyled list-unstyled-border" id="task-comments">
                    @foreach ($task->comments as $comment)
                        <li class="row media border-bottom mb-3">
                            <div class="col-1">
                                <img alt="image" data-bs-toggle="tooltip" data-bs-placement="top"
                                    title="{{ $comment->user->name }}"
                                    @if ($comment->user->avatar) src="{{ get_file($comment->user->avatar) }}" @else src="{{ get_file('avatar.png') }}" @endif
                                    class="mr-3 avatar-sm rounded-circle img-thumbnail"
                                    style="max-width: 30px; max-height: 30px;">
                            </div>
                            <div class="col media-body mb-2 top-10-scroll" style="max-height: 100px">
                                <h5 class="mt-0 mb-1 form-control-label">
                                    @if ($comment->user_type != 'Client')
                                        {{ $comment->user->name }}
                                    @else
                                        {{ $comment->client->name }}
                                    @endif
                                </h5>
                                {{ $comment->comment }}
                            </div>
                            @auth('web')
                                <div class="col-auto text-end row_line_style">
                                    @permission('task comment delete')
                                        <a href="#"
                                            class="action-btn btn-danger mt-1 btn btn-sm d-inline-flex align-items-center delete-comment"
                                            data-url="{{ route('comment.destroy', [$task->project_id, $task->id, $comment->id]) }}">
                                            <i class="ti ti-trash"></i>
                                        </a>
                                    @endpermission
                                </div>
                            @endauth
                        </li>
                    @endforeach
                </ul>
            </div>
            <div class="tab-pane fade" id="file-data" role="tabpanel" aria-labelledby="profile-tab">
                <div id="comments-file" class="mt-3">
                    @foreach ($task->taskFiles as $file)
                        <div class="card pb-0 mb-1 shadow-none border">
                            <div class="card-body p-3">
                                <div class="row align-items-center">
                                    <div class="col-auto">
                                        <div class="avatar-sm">
                                            <span class="avatar-title rounded text-uppercase">
                                                <img src="{{ get_file($file->file) }}" width="60px" height="60px"
                                                    alt="">
                                            </span>
                                        </div>
                                    </div>
                                    <div class="col pl-0">
                                        <a href="#" class="text-muted form-control-label">{{ $file->name }}</a>
                                        <p class="mb-0">{{ $file->file_size }}</p>
                                    </div>
                                    <div class="col-auto">
                                        <!-- Button -->
                                        <a download href="{{ get_file($file->file) }}"
                                            class="action-btn btn-primary btn btn-sm d-inline-flex align-items-center">
                                            <i class="ti ti-download"></i>
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    @endforeach
                </div>
            </div>
            <div class="tab-pane fade mt-3" id="sub-task-data" role="tabpanel" aria-labelledby="contact-tab">
                <ul class="list-group mt-3 " id="subtasks">
                    @foreach ($task->sub_tasks as $subTask)
                        <li class="list-group-item row">
                            <div class="form-check form-switch d-inline-block d-flex justify-content-between">
                                <div>
                                    <label class="custom-control-label form-control-label"
                                        for="option{{ $subTask->id }}">{{ $subTask->name }}</label>
                                </div>
                            </div>
                        </li>
                    @endforeach
                </ul>
            </div>
        </div>
    </div>
@else
    <div class="container mt-5">
        <div class="card">
            <div class="card-body p-4">
                <div class="page-error">
                    <div class="page-inner">
                        <h1>{{ __('404') }}</h1>
                        <div class="page-description">
                            {{ __('Page Not Found') }}
                        </div>
                        <div class="page-search">
                            <p class="text-muted mt-3">
                                {{ __("It's looking like you may have taken a wrong turn. Don't worry... it happens to the best of us. Here's a little tip that might help you get back on track.") }}
                            </p>
                            <div class="mt-3">
                                <a class="btn-return-home badge-blue" href="{{ route('home') }}"><i
                                        class="fas fa-reply"></i> {{ __('Return Home') }}</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
@endif
