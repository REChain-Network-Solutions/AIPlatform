@if ($currentWorkspace && $bug)
    <div class="modal-body">
        <div>
            <div class="form-control-label">{{ __('Description') }}:</div>
            <p class="text-muted mb-2">
                {{ $bug->description }}
            </p>

            @if (!empty($customFields) && count($bug->customField) > 0)
                <div class="pt-3">
                    <div class="row">
                        @foreach ($customFields as $field)
                            <div class="col-6">
                                <p>
                                    <strong class="text-dark">{{ $field->name }} :</strong>
                                    @if ($field->type == 'attachment')
                                        <a href="{{ get_file($bug->customField[$field->id]) }}" target="_blank" class="text-primary">
                                            <i class="ti ti-file"></i> {{ __('View File') }}
                                        </a>
                                    @else
                                        <span class="text-muted">
                                            {{ !empty($bug->customField[$field->id]) ? $bug->customField[$field->id] : '-' }}
                                        </span>
                                    @endif
                                </p>
                            </div>
                        @endforeach
                    </div>
                </div>
            @endif

            <div class="row mb-4">
                <div class="col-md-4">
                    <div class="form-control-label">{{ __('Create Date') }}</div>
                    <p class="mt-1">{{ company_date_formate($bug->created_at) }}</p>
                </div>
                <div class="col-md-3">
                    <div class="form-control-label">{{ __('Assigned') }}</div>
                    <img alt="image" data-bs-toggle="tooltip" data-bs-placement="top"
                        title="{{ !empty($bug->user->name) ? $bug->user->name : '' }}"
                        @if ($bug->user) src="{{ get_file($bug->user->avatar) }}" @else src="{{ get_file('uploads/users-avatar/avatar.png') }}" @endif
                        class="rounded-circle " width="30px" height="30px">
                </div>
            </div>

            <ul class="nav nav-pills mb-3" id="pills-tab" role="tablist"
                style="border: solid #eee;border-radius: 13px;">
                <li class="nav-item">
                    <a class=" nav-link active" id="comments-tab" data-bs-toggle="tab" href="#comments-data"
                        role="tab" aria-controls="home" aria-selected="false"> {{ __('Comments') }} </a>
                </li>
                @permission('bug file uploads')
                    <li class="nav-item">
                        <a class="nav-link" id="file-tab" data-bs-toggle="tab" href="#file-data" role="tab"
                            aria-controls="profile" aria-selected="false"> {{ __('Files') }} </a>
                    </li>
                @endpermission
            </ul>
            <div class="tab-content" id="myTabContent">
                <div class="tab-pane fade active show" id="comments-data" role="tabpanel"
                    aria-labelledby="comments-tab">
                    @permission('bug comments create')
                        <form method="post" id="form-comment"
                            data-action="{{ route('bug.comment.store', [$bug->project_id, $bug->id, $clientID]) }}">
                            <textarea class="form-control form-control-light mb-2" name="comment" placeholder="{{ __('Write message') }}"
                                id="example-textarea" rows="3" required></textarea>
                                <span class="text-danger comment-requried"></span>
                            <div class="text-end">
                                <div class="btn-group mt-3">
                                    <button type="button" class="btn btn btn-primary">{{ __('Submit') }}</button>
                                </div>
                            </div>
                        </form>
                    @endpermission
                    <ul class="list-unstyled list-unstyled-border" id="comments">
                        @foreach ($bug->comments as $comment)
                            <li class="row media border-bottom mb-3">
                                <div class="col-1">
                                    <img alt="image" data-bs-toggle="tooltip" data-bs-placement="top"
                                        title="{{ $comment->user->name }}"
                                        @if ($comment->user->avatar) src="{{ get_file($comment->user->avatar) }}" @else src="{{ get_file('avatar.png') }}" @endif
                                        class="mr-3 avatar-sm rounded-circle img-thumbnail" width=""
                                        style="max-width: 30px; max-height: 30px;">
                                </div>
                                <div class="col media-body mb-2 top-10-scroll" style="max-height: 100px">
                                    <h5 class="mt-0 mb-1 form-control-label">
                                        @if ($comment->user_type != 'client')
                                            {{ $comment->user->name }}
                                        @else
                                            {{ $comment->client->name }}
                                        @endif
                                    </h5>
                                    {{ $comment->comment }}
                                </div>
                                <div class="col-auto text-end mt-2">
                                    @auth('web')
                                        @permission('bug comments delete')
                                        <a href="#" class="action-btn btn-danger btn btn-sm d-inline-flex align-items-center delete-comment" title="{{ __('delete') }}" data-toggle="tooltip" data-url="{{ route('bug.comment.destroy', [$bug->project_id, $bug->id, $comment->id]) }}">
                                             <i class="ti ti-trash"></i>
                                         </a>
                                        @endpermission
                                    @endauth
                                </div>
                            </li>
                        @endforeach
                    </ul>
                </div>
                <div class="tab-pane fade" id="file-data" role="tabpanel" aria-labelledby="file-tab">
                    @permission('bug file uploads')
                        <form method="post" id="form-file" enctype="multipart/form-data"
                            data-url="{{ route('bug.comment.store.file', [$bug->project_id, $bug->id, $clientID]) }}">
                            @csrf

                            <div class="form-group col-md-12">
                                {{ Form::label('file', __('Image'), ['class' => 'form-label']) }}<x-required></x-required>
                                <div class="choose-file">
                                        <input type="file" class="form-control" name="file" id="file"
                                            data-filename="file" onchange="document.getElementById('blah').src = window.URL.createObjectURL(this.files[0])" style="width:365px;">
                                        <img id="blah" width="25%" class="mt-3">
                                    </label>
                                </div>
                            </div>

                            <div class="col-auto text-end">
                                <button type="submit" class="btn btn-primary">{{ __('Upload') }}</button>
                            </div>
                        </form>
                    @endpermission
                    <div id="comments-file" class="mt-3">
                        @foreach ($bug->bugFiles as $file)
                            <div class="card pb-0 mb-1 shadow-none border">
                                <div class="card-body p-3">
                                    <div class="row align-items-center">
                                        <div class="col-auto">
                                            <div class="avatar-sm">
                                                <span class="avatar-title rounded text-uppercase">
                                                    <img src="{{ get_file($file->file) }}" width="60px"
                                                        height="60px" alt="">
                                                </span>
                                            </div>
                                        </div>
                                        <div class="col pl-0">
                                            <a href="#"
                                                class="text-muted form-control-label">{{ $file->name }}</a>
                                            <p class="mb-0">{{ $file->file_size }}</p>
                                        </div>
                                        <div class="col-auto">
                                            <a download href="{{ get_file($file->file) }}"
                                                class="action-btn btn-primary  btn btn-sm d-inline-flex align-items-center me-1" title="{{ __('download') }}" data-toggle="tooltip">
                                                <i class="ti ti-download"></i>
                                            </a>
                                            @permission('bug file delete')
                                                <a href="#"
                                                    class="action-btn btn-danger  btn btn-sm d-inline-flex align-items-center delete-comment-file" title="{{ __('delete') }}" data-toggle="tooltip"
                                                    data-url="{{ route('bug.comment.destroy.file', [$bug->project_id, $bug->id, $file->id]) }}">
                                                    <i class="ti ti-trash"></i>
                                                </a>
                                            @endpermission
                                        </div>
                                    </div>
                                </div>
                            </div>
                        @endforeach
                    </div>
                </div>
            </div>
        </div>
    </div>
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
