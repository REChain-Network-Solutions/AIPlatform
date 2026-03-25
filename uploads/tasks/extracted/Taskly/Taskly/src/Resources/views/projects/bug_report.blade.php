@extends('layouts.main')

@section('page-title')
    {{__('Bug Report')}}
@endsection

@section('page-breadcrumb')
    {{ __('Project Details') }},
    {{ __("Bug Report") }}
@endsection
@push('css')
<link rel="stylesheet" href="{{ asset('assets/css/plugins/dragula.min.css')}}">
<style type="text/css">
    .hight_img{
      max-width: 30px !important;
       max-height: 30px !important;
    }
</style>
@endpush

@section('page-action')
<div class="d-flex">
    @stack('addButtonHook')
    <a href="{{ route('projectbug.list',[$project->id]) }}" class="btn btn-sm btn-primary me-2" data-bs-toggle="tooltip"title="{{ __('List View') }}">
        <i class="ti ti-list text-white"></i>
    </a>
    @permission('bug create')
    <a class="btn btn-sm btn-primary me-2" data-ajax-popup="true" data-size="lg" data-title="{{ __('Create Bug') }}" data-url="{{route('projects.bug.report.create',$project->id)}}" data-bs-toggle="tooltip" title="{{ __('Add Bug') }}"><i class="ti ti-plus"></i></a>
    @endpermission
    <a href="{{route('projects.show',[$project->id])}}" class="btn-submit btn btn-sm btn-primary" data-bs-toggle="tooltip" title="{{ __('Back') }}"><i class="ti ti-arrow-back-up"></i></a>
</div>

@endsection

@section('content')
        @if($project)
            <div class="row">
                <div class="col-sm-12">
                <div class="row kanban-wrapper horizontal-scroll-cards pt-3" data-toggle="dragula" data-containers='{{json_encode($statusClass)}}'>
                     @foreach($stages as $stage)
                    <div class="col" id="backlog">
                        <div class="card card-list">
                            <div class="card-header" >
                                <div class="float-end">
                                    <button class="btn-submit btn btn-md btn-primary btn-icon px-1  py-0">
                                       <span class="badge badge-secondary rounded-pill count">{{$stage->bugs->count()}}</span>
                                    </button>
                                </div>
                                <h4 class="mb-0" >{{$stage->name}}</h4>
                            </div>
                            <div id="{{'task-list-'.str_replace(' ','_',$stage->id)}}" data-status="{{$stage->id}}" class="card-body kanban-box">
                                 @foreach($stage->bugs as $bug)
                                <div class="card" id="{{$bug->id}}">
                                    <div class="position-absolute top-0 start-0 pt-3 ps-3">
                                      @if($bug->priority =='Low')
                                            <div class="badge bg-success p-2 px-3"> {{ $bug->priority }}</div>
                                        @elseif($bug->priority =='Medium')
                                            <div class="badge bg-warning p-2 px-3"> {{ $bug->priority }}</div>
                                        @elseif($bug->priority =='High')
                                            <div class="badge bg-danger p-2 px-3"> {{ $bug->priority }}</div>
                                        @endif
                                    </div>
                                    <div class="card-header border-0 pb-0 position-relative">
                                        <div style="padding: 30px 2px;">
                                            <a href="#" data-size="lg" data-url="{{route('projects.bug.report.show',[$bug->project_id,$bug->id])}}" data-ajax-popup="true" data-title="{{__('Bug Detail')}}" class="h6 task-title"><h5>{{$bug->title}}</h5>
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
                                                    @permission('bug show')
                                                    <a class="dropdown-item" data-ajax-popup="true" data-size="lg" data-title="{{ __('View Bug Report') }}" data-url="{{route('projects.bug.report.show',[$bug->project_id,$bug->id])}}"><i class="ti ti-eye me-1"></i>
                                                        {{__('View')}}</a>
                                                    @endpermission
                                                    @permission('bug edit')
                                                    <a class="dropdown-item" data-ajax-popup="true" data-size="lg" data-title="{{ __('Edit Bug Report') }}" data-url="{{route('projects.bug.report.edit',[$bug->project_id,$bug->id])}}"><i class="ti ti-pencil me-1"></i>
                                                        {{__('Edit')}}</a>
                                                    @endpermission
                                                    @permission('bug delete')
                                                    <form id="delete-form-{{$bug->id}}" action="{{ route('projects.bug.report.destroy',[$bug->project_id,$bug->id]) }}" method="POST">
                                                        @csrf
                                                        @method('DELETE')
                                                        <a href="#" class="dropdown-item bs-pass-para  text-danger show_confirm" data-confirm="{{__('Are You Sure?')}}" data-text="{{__('This action can not be undone. Do you want to continue?')}}"  data-confirm-yes="delete-form-{{$bug->id}}"><i class="ti ti-trash me-1"></i>
                                                            {{__('Delete')}}
                                                        </a>
                                                    </form>
                                                    @endpermission
                                                </div>

                                            </div>
                                        </div>
                                    </div>
                                    <div class="card-body pt-0">
                                        <div class="d-flex align-items-center justify-content-between">
                                            <ul class="list-inline mb-0">
                                                <li class="list-inline-item d-inline-flex align-items-center"><i class="f-16 text-primary ti ti-message-2"></i>
                                                    {{$bug->comments->count()}} {{ __('Comments') }}
                                                </li>
                                            </ul>

                                            <div class="user-group">
                                                <a href="#" class="img_group">
                                                    <img alt="image" data-bs-toggle="tooltip" data-bs-placement="top" title="{{!empty($bug->user->name)?$bug->user->name:''}}" @if($bug->user) src="{{get_file($bug->user->avatar)}}" @else src="{{ get_file('uploads/users-avatar/avatar.png')}}"  @endif class="rounded-circle " width="40px" height="40px">
                                                </a>
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
@endsection
@if($project)
    @push('scripts')
    <script src="{{asset('assets/js/plugins/dragula.min.js')}}"></script>
    <script src="{{ asset('js/letter.avatar.js') }}"></script>
        <!-- third party js -->
        <script>
            !function (a) {
                "use strict";
                var t = function () {
                    this.$body = a("body")
                };
                t.prototype.init = function () {
                    a('[data-toggle="dragula"]').each(function () {
                        var t = a(this).data("containers"), n = [];
                        if (t) for (var i = 0; i < t.length; i++) n.push(a("#" + t[i])[0]); else n = [a(this)[0]];
                        var r = a(this).data("handleclass");
                        r ? dragula(n, {
                            moves: function (a, t, n) {
                                return n.classList.contains(r)
                            }
                        }) : dragula(n).on('drop', function (el, target, source, sibling) {
                            var sort = [];
                            $("#" + target.id + " > div").each(function (key) {
                                sort[key] = $(this).attr('id');
                            });
                            var id = el.id;
                            var old_status = $("#" + source.id).data('status');
                            var new_status = $("#" + target.id).data('status');
                            var project_id = '{{$project->id}}';

                            $("#" + source.id).parents('.card-list').find('.count').text($("#" + source.id + " > div").length);
                            $("#" + target.id).parents('.card-list').find('.count').text($("#" + target.id + " > div").length);
                            $.ajax({
                                url:'{{route('projects.bug.report.update.order',[$project->id])}}',
                                type: 'POST',
                                data: {
                                    id: id,
                                    sort: sort,
                                    new_status: new_status,
                                    old_status: old_status,
                                    project_id: project_id,
                                },
                                success: function (data) {
                                }
                            });
                        });
                    })
                }, a.Dragula = new t, a.Dragula.Constructor = t
            }(window.jQuery), function (a) {
                "use strict";
                    a.Dragula.init()
            }(window.jQuery);
        </script>
        <!-- third party js ends -->
        <script>
            $(document).on('click', '#form-comment button', function (e) {
                var comment = $.trim($("#form-comment textarea[name='comment']").val());
                if (comment != '') {
                    $.ajax({
                        url: $("#form-comment").data('action'),
                        data: {comment: comment},
                        type: 'POST',
                        success: function (data) {
                            data = JSON.parse(data);

                            if (data.user_type == 'Client') {
                                var avatar = "avatar='" + data.client.name + "'";
                                var html = "<li class='row media border-bottom mb-3'>" +
                                    "                    <div class='col-1'>" +
                                    "                       <img class='mr-3 avatar-sm rounded-circle img-thumbnail hight_img' width='80' " + avatar +
                                    "                    </div>" +
                                    " alt='" + data.client.name + "'>" +
                                    "                    <div class='col media-body mb-2 top-10-scroll' style='max-height:100px;'>" +
                                    "                        <h5 class='mt-0 mb-1 form-control-label'>" + data.client.name + "</h5>" +
                                    "                        " + data.comment +
                                    "                    </div>" +
                                    "                </li>";
                            } else {
                                var avatar = (data.user.avatar) ? "src='{{asset('')}}/" + data.user.avatar + "'" : "avatar='" + data.user.name + "'";
                                var html = "<li class='row media border-bottom mb-3'>" +
                                    "                    <div class='col-1'>" +
                                    "                       <img class='mr-3 avatar-sm rounded-circle img-thumbnail hight_img' width='90' " + avatar +
                                    " alt='" + data.user.name + "'>" +
                                    "                    </div>" +
                                    "                    <div class='col media-body mb-2 top-10-scroll' style='max-height:100px;'>" +
                                    "                        <h5 class='mt-0 mb-1 form-control-label'>" + data.user.name + "</h5>" +
                                    "                        " + data.comment +
                                    "                    </div>" +
                                    "                    <div class='col-auto text-end mt-2'>" +
                                    "                               <a href='#' class='delete-icon delete-comment action-btn btn-danger  btn btn-sm d-inline-flex align-items-center' data-url='" + data.deleteUrl + "'>" +
                                    "                                   <i class='ti ti-trash'></i>" +
                                    "                               </a>" +
                                    "                    </div>" +
                                    "            </li>";
                            }
                            $("#comments").prepend(html);
                            LetterAvatar.transform();

                            $("#form-comment textarea[name='comment']").val('');
                            toastrs('{{__('Success')}}', '{{ __("Comment Added Successfully!")}}', 'success');

                        },
                        error: function (data) {
                            toastrs('{{__('Error')}}', '{{ __("Some Thing Is Wrong!")}}', 'error');
                        }
                    });
                } else {
                    $('.comment-requried').text('Please write comment!')

                }
            });
            $(document).on("click", ".delete-comment", function (e) {
            e.preventDefault(); // Prevent default action
            var btn = $(this);
            var deleteUrl = btn.attr('data-url');

            var form = $(this).closest("form");
            var title = $(this).attr("data-confirm");
            var text = $(this).attr("data-text");
            if (title == '' || title == undefined) {
                title = "Are you sure?";

            }
            if (text == '' || text == undefined) {
                text = "This action can not be undone. Do you want to continue?";

            }
            const swalWithBootstrapButtons = Swal.mixin({
                customClass: {
                    confirmButton: 'btn btn-success',
                    cancelButton: 'btn btn-danger'
                },
                buttonsStyling: false
            })
            swalWithBootstrapButtons.fire({
                title: title,
                text: text,
                icon: 'warning',
                showCancelButton: true,
                confirmButtonText: 'Yes',
                cancelButtonText: 'No',
                reverseButtons: true
            }).then((result) => {
                if (result.isConfirmed) {
                    $.ajax({
                        url: deleteUrl,
                        type: 'DELETE',
                        dataType: 'JSON',
                        success: function (data) {
                            toastrs('{{__('Success')}}', '{{ __("Comment Deleted Successfully!")}}', 'success');

                            btn.closest('.media').remove();
                        },
                        error: function (data) {
                            data = data.responseJSON;
                            if (data.message) {
                                toastr.error(data.message);
                            } else {
                                toastr.error('{{ __("Something went wrong!") }}');
                            }
                        }
                    });
                }
            });
        });

            $(document).on('submit', '#form-file', function (e) {
                e.preventDefault();

                $.ajax({
                    url: $("#form-file").data('url'),
                    type: 'POST',
                    data: new FormData(this),
                    dataType: 'JSON',
                    contentType: false,
                    cache: false,
                    processData: false,
                    success: function (data) {
                        toastrs('{{__('Success')}}', '{{ __("File Upload Successfully!")}}', 'success');

                        var delLink = '';
                        if (data.deleteUrl.length > 0) {
                            delLink = "<a href='#' class='delete-icon delete-comment-file action-btn btn-danger  btn btn-sm d-inline-flex align-items-center '  data-url='" + data.deleteUrl + "'>" +
                                "                                        <i class='ti ti-trash'></i>" +
                                "                                    </a>";
                        }

                        var html = "<div class='card mb-1 shadow-none border'>" +
                            "                        <div class='card-body p-3'>" +
                            "                            <div class='row align-items-center'>" +
                            "                                <div class='col-auto'>" +
                            "                                    <div class='avatar-sm'>" +
                            "                                        <span class='avatar-title rounded text-uppercase'>" +
                    "  <img src='{{asset('')}}/"+
                   data.file +
                   "' width='60px' height='60px' >"+
                   "                                        </span>" +
                            "                                    </div>" +
                            "                                </div>" +
                            "                                <div class='col pl-0'>" +
                            "                                    <a href='#' class='text-muted form-control-label'>" + data.name + "</a>" +
                            "                                    <p class='mb-0'>" + data.file_size + "</p>" +
                            "                                </div>" +
                            "                                <div class='col-auto'>" +
                            "                                    <a download href='{{asset('')}}/" + data.file + "' class='action-btn btn-primary  btn btn-sm d-inline-flex align-items-center mx-1'>" +
                            "                                        <i class='ti ti-download'></i>" +
                            "                                    </a>" +
                            delLink +
                            "                                </div>" +
                            "                            </div>" +
                            "                        </div>" +
                            "                    </div>";
                        $("#comments-file").prepend(html);
                    },
                    error: function (data) {
                        data = data.responseJSON;
                        if (data.message) {
                            toastrs('{{__('Error')}}', data.message, 'error');
                            $('#file-error').text(data.errors.file[0]).show();
                        } else {
                            toastrs('{{__('Error')}}', data.error, 'error');
                        }
                    }
                });
            });
            $(document).on("click", ".delete-comment-file", function () {

            var form = $(this).closest("form");
            var title = $(this).attr("data-confirm");
            var text = $(this).attr("data-text");
            if (title == '' || title == undefined) {
                title = "Are you sure?";

            }
            if (text == '' || text == undefined) {
                text = "This action can not be undone. Do you want to continue?";

            }
            const swalWithBootstrapButtons = Swal.mixin({
                customClass: {
                    confirmButton: 'btn btn-success',
                    cancelButton: 'btn btn-danger'
                },
                buttonsStyling: false
            })
            swalWithBootstrapButtons.fire({
                title: title,
                text: text,
                icon: 'warning',
                showCancelButton: true,
                confirmButtonText: 'Yes',
                cancelButtonText: 'No',
                reverseButtons: true
            }).then((result) => {
                if (result.isConfirmed) {
                    var btn = $(this);
                        $.ajax({
                            url: $(this).attr('data-url'),
                            type: 'DELETE',
                            dataType: 'JSON',
                            success: function (data) {
                                toastrs('{{__('Success')}}', '{{ __("File Deleted Successfully!")}}', 'success');
                                btn.closest('.border').remove();
                            },
                            error: function (data) {
                                data = data.responseJSON;
                                if (data.message) {
                                    toastrs('{{__('Error')}}', data.message, 'error');
                                } else {
                                    toastrs('{{__('Error')}}', data.message, 'error');
                                }
                            }
                        });
                    }
                });
            });
        </script>
    @endpush
@endif

