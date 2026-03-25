@extends('layouts.admin')
@section('page-title')
    {{ __('Task Board') }}
@endsection
@section('content')
    <section class="section">
        @if ($project && $currentWorkspace)
            <div class="row">
                <div class="col-12">
                    <div class="container-kanban">
                        <div class="kanban-board project-task-kanban-box" data-toggle="dragula"
                            data-containers='{{ json_encode($statusClass) }}'>
                            @foreach ($stages as $stage)
                                <div class="kanban-col px-0">
                                    <div class="card-list card-list-flush">
                                        <div class="card-list-title row align-items-center mb-3">
                                            <div class="col">
                                                <h6 class="mb-0 text-white text-sm">{{ $stage->name }}</h6>
                                            </div>
                                            <div class="col text-right">
                                                <span
                                                    class="badge badge-secondary rounded-pill count">{{ $stage->tasks->count() }}</span>
                                            </div>
                                        </div>
                                        <div id="{{ 'task-list-' . str_replace(' ', '_', $stage->id) }}"
                                            data-status="{{ $stage->id }}" class="card-list-body scrollbar-inner">
                                            @foreach ($stage->tasks as $task)
                                                <div class="card card-progress draggable-item border shadow-none mb-3"
                                                    id="{{ $task->id }}">
                                                    <div class="card-body">
                                                        <div class="row align-items-center">
                                                            <div class="col-6">
                                                                @if ($task->priority == 'Low')
                                                                    <div class="badge badge-pill badge-xs badge-success">
                                                                        {{ $task->priority }}</div>
                                                                @elseif($task->priority == 'Medium')
                                                                    <div class="badge badge-pill badge-xs badge-warning">
                                                                        {{ $task->priority }}</div>
                                                                @elseif($task->priority == 'High')
                                                                    <div class="badge badge-pill badge-xs badge-danger">
                                                                        {{ $task->priority }}</div>
                                                                @endif
                                                            </div>
                                                            <div class="col-6 text-right">
                                                            </div>
                                                        </div>
                                                        <div class="row">
                                                            <div class="col-12">
                                                                <a href="#"
                                                                    data-url="@auth('web'){{ route('tasks.show', [$currentWorkspace->slug, $task->project_id, $task->id]) }}@elseauth{{ route('client.tasks.show', [$currentWorkspace->slug, $task->project_id, $task->id]) }}@endauth"
                                                                    data-ajax-popup="true"
                                                                    data-title="{{ __('Task Detail') }}"
                                                                    class="h6 task-title">{{ $task->title }}</a>
                                                            </div>
                                                        </div>
                                                        <div class="row">
                                                            <div class="col">
                                                                <div class="action-item">
                                                                    {{ \App\Models\Utility::dateFormat($task->start_date) }}
                                                                </div>
                                                            </div>
                                                            <div class="col text-right">
                                                                <div class="action-item">
                                                                    {{ \App\Models\Utility::dateFormat($task->due_date) }}
                                                                </div>
                                                            </div>
                                                        </div>
                                                        <div class="row">
                                                            <div class="col">
                                                                <div class="action-item">
                                                                    {{ $task->taskCompleteSubTaskCount() }}/{{ $task->taskTotalSubTaskCount() }}
                                                                </div>
                                                            </div>
                                                            <div class="col text-right">
                                                                <div class="avatar-group">
                                                                    @if ($users = $task->users())
                                                                        @foreach ($users as $key => $user)
                                                                            @if ($key < 3)
                                                                                <a href="#"
                                                                                    class="avatar rounded-circle avatar-sm">
                                                                                    <img alt="image"
                                                                                        data-toggle="tooltip"
                                                                                        data-original-title="{{ $user->name }}"
                                                                                        @if ($user->avatar) src="{{ asset('/storage/avatars/' . $user->avatar) }}" @else avatar="{{ $user->name }}" @endif>
                                                                                </a>
                                                                            @endif
                                                                        @endforeach
                                                                        @if (count($users) > 3)
                                                                            <a href="#"
                                                                                class="avatar rounded-circle avatar-sm">
                                                                                <img alt="image" data-toggle="tooltip"
                                                                                    data-original-title="{{ count($users) - 3 }} {{ __('more') }}"
                                                                                    avatar="+ {{ count($users) - 3 }}">
                                                                            </a>
                                                                        @endif
                                                                    @endif
                                                                </div>
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
                    </div>
                </div> <!-- end col -->
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
    </section>

@endsection
@if ($project && $currentWorkspace)
    @push('scripts')
        <!-- third party js -->
        <script src="{{ asset('custom/js/dragula.min.js') }}"></script>
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
                                url: '{{ route('tasks.update.order', [$currantWorkspace->slug, $project->id]) }}',
                                type: 'PUT',
                                data: {
                                    id: id,
                                    sort: sort,
                                    client_id: {{ $clientID }},
                                    new_status: new_status,
                                    old_status: old_status,
                                    project_id: project_id,
                                },
                                success: function(data) {}
                            });
                        });
                    })
                }, a.Dragula = new t, a.Dragula.Constructor = t
            }(window.jQuery),
            function(a) {
                "use strict";
                a.Dragula.init()
            }(window.jQuery);
        </script>
        <!-- third party js ends -->
        <script>
            $(document).on('click', '#form-comment button', function(e) {
                var comment = $.trim($("#form-comment textarea[name='comment']").val());
                if (comment != '') {
                    $.ajax({
                        url: $("#form-comment").data('action'),
                        data: {
                            comment: comment
                        },
                        type: 'POST',
                        success: function(data) {
                            data = JSON.parse(data);

                            if (data.user_type == 'Client') {
                                var avatar = "avatar='" + data.client.name + "'";
                                var html = "<li class='media'>" +
                                    "                    <img class='mr-3 avatar-sm rounded-circle img-thumbnail' width='60' " +
                                    avatar + " alt='" + data.client.name + "'>" +
                                    "                    <div class='media-body'>" +
                                    "                        <h5 class='mt-0'>" + data.client.name +
                                    "</h5>" +
                                    "                        " + data.comment +
                                    "                    </div>" +
                                    "                </li>";
                            } else {
                                var avatar = (data.user.avatar) ?
                                    "src='{{ asset('/storage/avatars/') }}/" + data.user.avatar + "'" :
                                    "avatar='" + data.user.name + "'";
                                var html = "<li class='media'>" +
                                    "                    <img class='mr-3 avatar-sm rounded-circle img-thumbnail' width='60' " +
                                    avatar + " alt='" + data.user.name + "'>" +
                                    "                    <div class='media-body'>" +
                                    "                        <h5 class='mt-0'>" + data.user.name + "</h5>" +
                                    "                        " + data.comment +
                                    "                    </div>" +
                                    "                </li>";
                            }

                            $("#comments").prepend(html);
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

            $(document).on('submit', '#form-file', function(e) {
                e.preventDefault();
                $.ajax({
                    url: $("#form-file").data('action'),
                    type: 'POST',
                    data: new FormData(this),
                    dataType: 'JSON',
                    contentType: false,
                    cache: false,
                    processData: false,
                    success: function(data) {
                        toastrs('{{ __('Success') }}', '{{ __('Comment Added Successfully!') }}',
                            'success');
                        var delLink = '';

                        if (data.deleteUrl.length > 0) {
                            delLink =
                                "<a href='#' class='text-danger text-muted delete-comment-file'  data-url='" +
                                data.deleteUrl + "'>" +
                                "                                        <i class='fas fa-trash'></i>" +
                                "                                    </a>";
                        }

                        var html = "<div class='card mb-1 shadow-none border'>" +
                            "                        <div class='card-body py-2'>" +
                            "                            <div class='row align-items-center'>" +
                            "                                <div class='col-auto'>" +
                            "                                    <div class='avatar-sm'>" +
                            "                                        <span class='avatar-title rounded text-uppercase'>" +
                            data.extension +
                            "                                        </span>" +
                            "                                    </div>" +
                            "                                </div>" +
                            "                                <div class='col pl-0'>" +
                            "                                    <a href='#' class='text-muted font-weight-bold'>" +
                            data.name + "</a>" +
                            "                                    <p class='mb-0'>" + data.file_size +
                            "</p>" +
                            "                                </div>" +
                            "                                <div class='col-auto'>" +
                            "                                    <a download href='{{ asset('/storage/tasks/') }}/" +
                            data.file + "' class='btn btn-link text-muted'>" +
                            "                                        <i class='dripicons-download'></i>" +
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
                            toastrs('{{ __('Error') }}', '{{ __('Some Thing Is Wrong!') }}', 'error');
                        }
                    }
                });
            });
        </script>
    @endpush
@endif
