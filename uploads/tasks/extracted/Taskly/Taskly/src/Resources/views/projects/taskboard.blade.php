@extends('layouts.main')
@section('page-title')
    {{ __('Task Board') }}
@endsection
@section('page-breadcrumb')
    {{ __('Project Details') }},{{ __('Task Board') }}
@endsection
@section('page-action')
    <div class="d-flex">
        @stack('addButtonHook')
        <a href="{{ route('projects.calendar', [$project->id]) }}" class="btn btn-sm btn-primary me-2" data-bs-toggle="tooltip"
            data-bs-original-title="{{ __('Calendar View') }}" data-bs-original-title="{{ __('Calendar View') }}">
            <i class="ti ti-calendar"></i>
        </a>
        <a href="{{ route('projects.gantt', [$project->id]) }}" class="btn btn-sm btn-primary me-2" data-bs-toggle="tooltip"
            data-bs-original-title="{{ __('Time Line') }}">
            <i class="ti ti-alarm"></i>
        </a>
        <a href="{{ route('projecttask.list', [$project->id]) }}" class="btn btn-sm btn-primary me-2"
            data-bs-toggle="tooltip" data-bs-original-title="{{ __('List View') }}" title="{{ __('List View') }}">
            <i class="ti ti-list text-white"></i>
        </a>
        @permission('task create')
            <a class="btn btn-sm btn-primary me-2" data-ajax-popup="true" data-size="lg" data-title="{{ __('Create Task Board') }}"
                data-url="{{ route('tasks.create', [$project->id]) }}" data-bs-toggle="tooltip"
                data-bs-original-title="{{ __('Create') }}"><i class="ti ti-plus"></i></a>
        @endpermission
        <a href="{{ route('projects.show', [$project->id]) }}" class="btn-submit btn btn-sm btn-primary"
            data-bs-toggle="tooltip" data-bs-original-title="{{ __('Back') }}">
            <i class=" ti ti-arrow-back-up"></i>
        </a>
    </div>
@endsection
@push('css')
    @if (module_is_active('Signature'))
        <link rel="stylesheet" href="{{ asset('packages/workdo/Signature/src/Resources/assets/css/custom.css') }}">
    @endif
    <link rel="stylesheet" href="{{ asset('assets/css/plugins/dragula.min.css') }}">
@endpush
@php
    $UsersData = \Workdo\Taskly\Entities\Task::getUsersData();
@endphp
@section('content')
    @if ($project && $currentWorkspace && Laratrust::hasPermission('task manage'))
        <div class="row">
            <div class="col-sm-12">
                <div class="row kanban-wrapper horizontal-scroll-cards pt-3" data-toggle="dragula"
                    data-containers='{{ json_encode($statusClass) }}'>
                    @foreach ($stages as $stage)
                        <div class="col" id="backlog">
                            <div class="card card-list">
                                <div class="card-header">
                                    <div class="float-end">
                                        <button class="btn-submit btn btn-md btn-primary btn-icon px-1  py-0">
                                            <span
                                                class="badge badge-secondary rounded-pill count">{{ $stage->tasks->count() }}</span>
                                        </button>
                                    </div>
                                    <h4 class="mb-0">{{ $stage->name }}</h4>
                                </div>
                                <div id="{{ 'task-list-' . str_replace(' ', '_', $stage->id) }}"
                                    data-status="{{ $stage->id }}" class="card-body kanban-box">
                                    @foreach ($stage->tasks as $task)
                                        <div class="card" id="{{ $task->id }}">
                                            <div class="position-absolute top-0 start-0 pt-3 ps-3">
                                                @if ($task->priority == 'Low')
                                                    <div class="badge bg-success p-2 px-3"> {{ $task->priority }}
                                                    </div>
                                                @elseif($task->priority == 'Medium')
                                                    <div class="badge bg-warning p-2 px-3"> {{ $task->priority }}
                                                    </div>
                                                @elseif($task->priority == 'High')
                                                    <div class="badge bg-danger p-2 px-3"> {{ $task->priority }}
                                                    </div>
                                                @endif
                                            </div>
                                            <div class="card-header border-0 pb-0 position-relative">
                                                <div style="padding: 30px 2px;">
                                                    <a href="#"
                                                        data-url="{{ route('tasks.show', [$task->project_id, $task->id]) }}"
                                                        data-size="lg" data-ajax-popup="true"
                                                        data-title="{{ __('View Task Board') }}" class="h6 task-title">
                                                        <h5>{{ $task->title }}</h5>
                                                        <br>
                                                        <small>{{ !empty($task->milestones) ? $task->milestones->title : '' }}</small>

                                                    </a>
                                                </div>
                                                <div class="card-header-right">

                                                    <div class="btn-group card-option">
                                                        <button type="button" class="btn dropdown-toggle"
                                                            data-bs-toggle="dropdown" aria-haspopup="true"
                                                            aria-expanded="false">
                                                            <i class="feather icon-more-vertical"></i>
                                                        </button>
                                                        <div class="dropdown-menu dropdown-menu-end">
                                                            @permission('task show')
                                                                <a class="dropdown-item" data-ajax-popup="true" data-size="lg"
                                                                    data-title="{{ __('View Task Board') }}"
                                                                    data-url="{{ route('tasks.show', [$task->project_id, $task->id]) }}"><i
                                                                        class="ti ti-eye me-1"></i> {{ __('View') }}</a>
                                                            @endpermission
                                                            @permission('task edit')
                                                                <a class="dropdown-item" data-ajax-popup="true" data-size="lg"
                                                                    data-title="{{ __('Edit Task Board') }}"data-url="{{ route('tasks.edit', [$task->project_id, $task->id]) }}"><i
                                                                        class="ti ti-pencil me-1"></i> {{ __('Edit') }}</a>
                                                            @endpermission
                                                            @permission('task delete')
                                                                <form id="delete-form-{{ $task->id }}"
                                                                    action="{{ route('tasks.destroy', [$task->project_id, $task->id]) }}"
                                                                    method="POST">
                                                                    @csrf
                                                                    @method('DELETE')
                                                                    <a href="#"
                                                                        class="dropdown-item bs-pass-para text-danger show_confirm"
                                                                        data-confirm="{{ __('Are You Sure?') }}"
                                                                        data-text="{{ __('This action can not be undone. Do you want to continue?') }}"
                                                                        data-confirm-yes="delete-form-{{ $task->id }}"> <i
                                                                            class="ti ti-trash me-1"></i>
                                                                        {{ __('Delete') }}
                                                                    </a>
                                                                </form>
                                                            @endpermission
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            <div class="card-body pt-0">
                                                <div class="row">
                                                    <div class="col">
                                                        <div class="action-item">
                                                            {{ company_date_formate($task->start_date) }}
                                                        </div>
                                                    </div>
                                                    <div class="col text-end">
                                                        <div class="action-item">
                                                            {{ company_date_formate($task->due_date) }}
                                                        </div>
                                                    </div>
                                                </div>
                                                <div class="d-flex align-items-center justify-content-between">
                                                    <ul class="list-inline mb-0">
                                                        <li class="list-inline-item d-inline-flex align-items-center"><i
                                                                class="f-16 text-primary ti ti-brand-telegram"></i>
                                                            {{ $task->taskCompleteSubTaskCount() }}/{{ $task->taskTotalSubTaskCount() }}
                                                        </li>
                                                    </ul>
                                                    <div class="user-group">
                                                        @if (count($UsersData) > 0)
                                                            @foreach (explode(',', $task->assign_to) as $user)
                                                                {{-- @if ($key < 3) --}}
                                                                <img alt="image" data-bs-toggle="tooltip"
                                                                    data-bs-placement="top"
                                                                    title="{{ isset($UsersData[$user]['name']) ? $UsersData[$user]['name'] : '' }}"
                                                                    @if (isset($UsersData[$user]['avatar'])) src="{{ get_file($UsersData[$user]['avatar']) }}" @else src="{{ get_file('avatar.png') }}" @endif
                                                                    class="rounded-circle " width="40px"
                                                                    height="40px">
                                                                {{-- @endif --}}
                                                            @endforeach
                                                            @if (count($UsersData) > 3)
                                                                <img alt="image" data-toggle="tooltip"
                                                                    data-original-title="{{ count($UsersData) - 3 }} {{ __('more') }}"
                                                                    avatar="+ {{ count($UsersData) - 3 }}">
                                                            @endif
                                                        @endif
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    @endforeach
                                    <span class="empty-container" data-placeholder="Empty"></span>
                                </div>
                            </div>
                        </div>
                    @endforeach
                </div>
                <!-- [ sample-page ] end -->
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
@endsection
@if ($project && $currentWorkspace)
    @push('scripts')
        <!-- third party js -->
        <script src="{{ asset('assets/js/plugins/dragula.min.js') }}"></script>
        <script src="{{ asset('js/letter.avatar.js') }}"></script>
        <script>
            ! function(a) {
                "use strict";
                var t = function() {
                    this.$body = a("body")
                };
                t.prototype.init = function() {
                    a('[data-toggle="dragula"]').each(function() {
                        var t = a(this).data("containers"),
                            n = [];
                        if (t)
                            for (var i = 0; i < t.length; i++) n.push(a("#" + t[i])[0]);
                        else n = [a(this)[0]];
                        var r = a(this).data("handleclass");
                        r ? dragula(n, {
                            moves: function(a, t, n) {
                                return n.classList.contains(r)
                            }
                        }) : dragula(n).on('drop', function(el, target, source, sibling) {
                            var sort = [];
                            $("#" + target.id + " > div").each(function(key) {
                                sort[key] = $(this).attr('id');
                            });
                            var id = el.id;
                            var old_status = $("#" + source.id).data('status');
                            var new_status = $("#" + target.id).data('status');
                            var project_id = '{{ $project->id }}';

                            $("#" + source.id).parents('.card-list').find('.count').text($("#" + source.id +
                                " > div").length);
                            $("#" + target.id).parents('.card-list').find('.count').text($("#" + target.id +
                                " > div").length);
                            $.ajax({
                                url: '{{ route('tasks.update.order', [$project->id]) }}',
                                type: 'POST',
                                data: {
                                    id: id,
                                    sort: sort,
                                    new_status: new_status,
                                    old_status: old_status,
                                    project_id: project_id,
                                    _token: "{{ csrf_token() }}"
                                },
                                success: function(data) {}
                            });
                        });
                    })
                }, a.Dragula = new t, a.Dragula.Constructor = t
            }(window.jQuery),
            function(a) {
                "use strict";
                @permission('task move')
                    a.Dragula.init();
                @endpermission ()
            }(window.jQuery);
        </script>
        <!-- third party js ends -->
        <script>
            $(document).on('submit', '#form-comment', function(e) {
                e.preventDefault();
                var comment = $.trim($("#form-comment textarea[name='comment']").val());
                if (comment != '') {
                    $.ajax({
                        url: $("#form-comment").data('action'),
                        data: {
                            comment: comment,
                            _token: "{{ csrf_token() }}"
                        },
                        type: 'POST',
                        success: function(data) {
                            data = JSON.parse(data);

                            if (data.user_type == 'Client') {
                                var avatar = "avatar='" + data.client.name + "'";
                                var html = "<li class='media border-bottom mb-3'>" +
                                    "                    <img class='mr-3 avatar-sm rounded-circle img-thumbnail hight_img' width='60' " +
                                    avatar + " alt='" + data.client.name + "'>" +
                                    "                    <div class='media-body mb-2'>" +
                                    "                    <div class='float-left'>" +
                                    "                        <h5 class='mt-0 mb-1 form-control-label'>" +
                                    data.client.name + "</h5>" +
                                    "                        " + data.comment +
                                    "                    </div>" +
                                    "                    </div>" +
                                    "                </li>";
                            } else {
                                var avatar = (data.user.avatar) ?
                                    "src='{{ asset('') }}" + data.user.avatar + "'" :
                                    "avatar='" + data.user.name + "'";
                                var html = "<li class='row media border-bottom mb-3'>" +
                                    "                    <div class='col-1'>" +
                                    "                        <img class='mr-3 avatar-sm rounded-circle img-thumbnail hight_img ' style='max-width: 30px; max-height: 30px;' " +
                                    avatar + " alt='" + data.user.name + "'>" +
                                    "                    </div>" +
                                    "                    <div class='col media-body mb-2 top-10-scroll' style='max-height:100px;'>" +
                                    "                        <h5 class='mt-0 mb-1 form-control-label'>" +
                                    data.user.name + "</h5>" +
                                    "                        " + data.comment +
                                    "                    </div>" +
                                    "                    <div class='col-auto text-end row_line_style'>" +
                                    "                           <a href='#' class='delete-icon action-btn btn-danger mt-1 btn btn-sm d-inline-flex align-items-center delete-comment' data-toggle='tooltip' title='{{ __('Delete') }}' data-url='" +
                                    data.deleteUrl + "'>" +
                                    "                               <i class='ti ti-trash'></i>" +
                                    "                           </a>" +
                                    "                     </div>" +
                                    "                </li>";
                            }

                            $("#task-comments").prepend(html);
                            LetterAvatar.transform();
                            $("#form-comment textarea[name='comment']").val('');
                            toastrs('{{ __('Success') }}', '{{ __('Comment Added Successfully!') }}',
                                'success');
                        },
                        error: function(data) {
                            toastrs('{{ __('Error') }}', '{{ __('Some Thing Is Wrong!') }}', 'error');
                        }
                    });
                } else {
                    toastrs('{{ __('Error') }}', '{{ __('Please write comment!') }}', 'error');
                }
            });
            $(document).on("click", ".delete-comment", function() {
                if (confirm('{{ __('Are you sure ?') }}')) {
                    var btn = $(this);
                    $.ajax({
                        url: $(this).attr('data-url'),
                        type: 'DELETE',
                        dataType: 'JSON',
                        success: function(data) {
                            toastrs('{{ __('Success') }}', '{{ __('Comment Deleted Successfully!') }}',
                                'success');
                            btn.closest('.media').remove();
                        },
                        error: function(data) {
                            data = data.responseJSON;
                            if (data.message) {
                                toastrs('{{ __('Error') }}', data.message, 'error');
                            } else {
                                toastrs('{{ __('Error') }}', '{{ __('Some Thing Is Wrong!') }}',
                                    'error');
                            }
                        }
                    });
                }
            });

            $(document).on('click', '#form-subtask button', function(e) {
                e.preventDefault();

                var name = $.trim($("#form-subtask input[name=name]").val());
                var due_date = $.trim($("#form-subtask input[name=due_date]").val());
                if (name == '' || due_date == '') {
                    toastrs('{{ __('Error') }}', '{{ __('Please enter fields!') }}', 'error');
                    return false;
                }

                $.ajax({
                    url: $("#form-subtask").data('action'),
                    type: 'POST',
                    data: {
                        name: name,
                        due_date: due_date,
                    },
                    dataType: 'JSON',
                    success: function(data) {
                        toastrs('{{ __('Success') }}', '{{ __('Sub Task Added Successfully!') }}',
                            'success');

                        var html = '<li class="list-group-item row">' +
                            '    <div class="form-check form-switch d-inline-block d-flex justify-content-between">' +
                            '       <div> <input type="checkbox" class="form-check-input" name="option" id="option' +
                            data.id + '" value="' + data.id + '" data-url="' + data.updateUrl + '">' +
                            '        <label class="custom-control-label form-control-label" for="option' +
                            data.id + '">' + data.name + '</label>' +
                            '    </div>' +
                            '    <div>' +
                            '        <a href="#" class=" action-btn btn-danger  btn btn-sm d-inline-flex align-items-center delete-icon delete-subtask" data-toggle="tooltip" title="{{ __('Delete') }}"  data-url="' +
                            data.deleteUrl + '">' +
                            '            <i class="ti ti-trash"></i>' +
                            '        </a>' +
                            '</div>' +
                            '    </div>' +
                            '</li>';

                        $("#subtasks").prepend(html);
                        $("#form-subtask input[name=name]").val('');
                        $("#form-subtask input[name=due_date]").val('');
                        $("#form-subtask").collapse('toggle');
                    },
                    error: function(data) {
                        data = data.responseJSON;
                        if (data.message) {
                            toastrs('{{ __('Error') }}', data.message, 'error');
                            $('#file-error').text(data.errors.file[0]).show();
                        } else {
                            toastrs('{{ __('Error') }}', '{{ __('Some Thing Is Wrong!') }}', 'error');
                        }
                    }
                });
            });
            $(document).on("change", "#subtasks input[type=checkbox]", function() {
                $.ajax({
                    url: $(this).attr('data-url'),
                    type: 'POST',
                    data: {
                        _token: "{{ csrf_token() }}"
                    },
                    dataType: 'JSON',
                    success: function(data) {
                        toastrs('{{ __('Success') }}', '{{ __('Subtask Updated Successfully!') }}',
                            'success');
                    },
                    error: function(data) {
                        data = data.responseJSON;
                        if (data.message) {
                            toastrs('{{ __('Error') }}', data.message, 'error');
                        } else {
                            toastrs('{{ __('Error') }}', '{{ __('Some Thing Is Wrong!') }}', 'error');
                        }
                    }
                });
            });
            $(document).on("click", ".delete-subtask", function() {
                if (confirm('{{ __('Are you sure ?') }}')) {
                    var btn = $(this);
                    $.ajax({
                        url: $(this).attr('data-url'),
                        type: 'DELETE',
                        dataType: 'JSON',
                        success: function(data) {
                            toastrs('{{ __('Success') }}', '{{ __('Subtask Deleted Successfully!') }}',
                                'success');
                            btn.closest('.list-group-item').remove();
                        },
                        error: function(data) {
                            data = data.responseJSON;
                            if (data.message) {
                                toastrs('{{ __('Error') }}', data.message, 'error');
                            } else {
                                toastrs('{{ __('Error') }}', '{{ __('Some Thing Is Wrong!') }}',
                                    'error');
                            }
                        }
                    });
                }
            });
            $(document).on('submit', '#form-file', function(e) {
                e.preventDefault();

                $.ajax({
                    url: $("#form-file").data('url'),
                    type: 'POST',
                    data: new FormData(this),
                    dataType: 'JSON',
                    contentType: false,
                    cache: false,
                    processData: false,
                    success: function(data) {
                        toastrs('Success', '{{ __('File Upload Successfully!') }}', 'success');

                        var delLink = '';
                        if (data.deleteUrl.length > 0) {
                            delLink =
                                "<a href='#' class=' action-btn btn-danger  btn btn-sm d-inline-flex align-items-center delete-icon delete-comment-file' data-toggle='tooltip' title='{{ __('Delete') }}'  data-url='" +
                                data.deleteUrl + "'>" +
                                "                                        <i class='ti ti-trash'></i>" +
                                "                                    </a>";
                        }

                        var html = "<div class='card mb-1 shadow-none border'>" +
                            "                        <div class='card-body p-3'>" +
                            "                            <div class='row align-items-center'>" +
                            "                                <div class='col-auto'>" +
                            "                                    <div class='avatar-sm'>" +
                            "                                        <span class='avatar-title rounded text-uppercase'>" +
                            " <img src='{{ asset('') }}" +
                            data.file +
                            "' width='60px' height='60px' >" +
                            "                                        </span>" +
                            "                                    </div>" +
                            "                                </div>" +
                            "                                <div class='col pl-0'>" +
                            "                                    <a href='#' class='text-muted form-control-label'>" +
                            data.name + "</a>" +
                            "                                    <p class='mb-0'></p>" +
                            "                                </div>" +
                            "                                <div class='col-auto'>" +
                            "                                    <a download href='{{ asset('') }}/" +
                            data.file +
                            "' class='edit-icon action-btn btn-primary  btn btn-sm d-inline-flex align-items-center mx-1' data-toggle='tooltip' title='{{ __('Download') }}'>" +
                            "                                        <i class='ti ti-download'></i>" +
                            "                                    </a>" +
                            delLink +
                            "                                </div>" +
                            "                            </div>" +
                            "                        </div>" +
                            "                    </div>";
                        $("#comments-file").prepend(html);
                    },
                    error: function(data) {
                        data = data.responseJSON;
                        if (data.message) {
                            toastrs('{{ __('Error') }}', data.message, 'error');
                            $('#file-error').text(data.errors.file[0]).show();
                        } else {
                            toastrs('{{ __('Error') }}', data.error, 'error');
                        }
                    }
                });
            });
            $(document).on("click", ".delete-comment-file", function() {
                if (confirm('{{ __('Are you sure ?') }}')) {
                    var btn = $(this);
                    $.ajax({
                        url: $(this).attr('data-url'),
                        type: 'DELETE',
                        data: {
                            _token: "{{ csrf_token() }}"
                        },
                        dataType: 'JSON',
                        success: function(data) {
                            toastrs('{{ __('Success') }}', '{{ __('File Deleted Successfully!') }}',
                                'success');
                            btn.closest('.border').remove();
                        },
                        error: function(data) {
                            data = data.responseJSON;
                            if (data.message) {
                                toastrs('{{ __('Error') }}', data.message, 'error');
                            } else {
                                toastrs('{{ __('Error') }}', '{{ __('Some Thing Is Wrong!') }}',
                                    'error');
                            }
                        }
                    });
                }
            });
        </script>
    @endpush
@endif
