@extends('layouts.app')

@push('styles')
    <link rel="stylesheet" href="{{ asset('vendor/css/dropzone.min.css') }}">
    <link rel="stylesheet" href="{{ asset('vendor/css/tagify.css') }}">
    <style>
        .message-action {
            visibility: hidden;
        }

        .ticket-left .card:hover .message-action {
            visibility: visible;
        }

        .file-action {
            visibility: hidden;
        }

        .file-card:hover .file-action {
            visibility: visible;
        }

        .frappe-chart .chart-legend {
            display: none;
        }
    </style>
    <script src="{{ asset('vendor/jquery/frappe-charts.min.iife.js') }}"></script>
    <script src="{{ asset('vendor/jquery/Chart.min.js') }}"></script>
@endpush

@php
    $editInspectionPermission = user()->permission('edit_quality_control');
    $deleteInspectionPermission = user()->permission('delete_quality_control');
@endphp

@section('filter-section')
    <div class="d-flex px-4 filter-box bg-white">
        <!-- HEADER START -->
        <a href="javascript:;"
            class="d-flex align-items-center height-44 text-dark-grey text-capitalize border-right-grey pr-3 reply-button"><i
                class="fa fa-reply mr-0 mr-lg-2 mr-md-2"></i><span
                class="d-none d-lg-block d-md-block">@lang('app.reply')</span></a>

        <div id="ticket-closed" @if ($schedule->status == 'closed') style="display:none" @endif>
            <a href="javascript:;" data-status="closed"
                class="d-flex align-items-center height-44 text-dark-grey text-capitalize border-right-grey px-3 submit-ticket"><i
                    class="fa fa-times-circle mr-0 mr-lg-2 mr-md-2"></i><span
                    class="d-none d-lg-block d-md-block">@lang('app.close')</span></a>
        </div>

        @if ($deleteInspectionPermission == 'all')
            <a href="javascript:;"
                class="d-flex align-items-center height-44 text-dark-grey text-capitalize border-right-grey px-3 delete-ticket"><i
                    class="fa fa-trash mr-0 mr-lg-2 mr-md-2"></i><span
                    class="d-none d-lg-block d-md-block">@lang('app.delete')</span>
            </a>
        @endif

        <a onclick="openTicketsSidebar()"
            class="d-flex d-lg-none ml-auto align-items-center justify-content-center height-44 text-dark-grey text-capitalize border-left-grey pl-3"><i
                class="fa fa-ellipsis-v"></i></a>
    </div>
    <!-- HEADER END -->
@endsection

@section('content')
@include('quality_control::partials.titan-links')

    <div class="ticket-wrapper bg-white border-top-0 d-lg-flex">
        <!-- LEFT START -->
        <div class="ticket-left w-100">
            <x-form id="updateTicket2" method="PUT">
                <input type="hidden" name="status" id="status" value="{{ $schedule->status }}">
                <input type="hidden" id="schedule_reply_id" value="">
                <!-- START -->
                <div class="d-flex justify-content-between align-items-center p-3 border-right-grey border-bottom-grey">
                    <span>
                        <p class="f-15 f-w-500 mb-0">{{ $schedule->subject }}</p>
                        <p class="f-11 text-lightest mb-0">Issued On
                            {{ $schedule->created_at->timezone(company()->timezone)->translatedFormat(company()->date_format . ' ' . company()->time_format) }}
                        </p>
                    </span>
                    <span>
                        @if ($schedule->status == 'open')
                            @php
                                $statusColor = 'yellow';
                            @endphp
                        @elseif($schedule->status == 'pending')
                            @php
                                $statusColor = 'red';
                            @endphp
                        @elseif($schedule->status == 'resolved')
                            @php
                                $statusColor = 'dark-green';
                            @endphp
                        @elseif($schedule->status == 'closed')
                            @php
                                $statusColor = 'blue';
                            @endphp
                        @endif
                        <p class="mb-0 text-capitalize ticket-status">
                            <x-status :color="$statusColor" :value="__('app.' . $schedule->status)" />
                        </p>
                    </span>

                </div>

                <!-- MESSAGE START -->
                <div class="ticket-msg border-right-grey" data-menu-vertical="1" data-menu-scroll="1"
                    data-menu-dropdown-timeout="500" id="ticketMsg">

                    @foreach ($schedule->reply as $reply)
                        {{-- <x-cards.ticket :message="$reply" :user="$reply->user" /> --}}
                        <div class="card ticket-message rounded-0 border-0  @if (user()->id == $user->id) bg-white-shade @endif"
                            id="message-{{ $reply->id }}">
                            <div class="card-horizontal">
                                <div class="card-img">
                                    <a
                                        href="{{ !is_null($user->employeeDetail) ? route('employees.show', $user->id) : route('clients.show', $user->id) }}"><img
                                            class="" src="{{ $user->image_url }}" alt="{{ $user->name }}"></a>
                                </div>
                                <div class="card-body border-0 pl-0">
                                    <div class="d-flex">
                                        <a
                                            href="{{ !is_null($user->employeeDetail) ? route('employees.show', $user->id) : route('clients.show', $user->id) }}">
                                            <h4 class="card-title f-13 f-w-500 text-dark mr-3">{{ $user->name }}</h4>
                                        </a>
                                        <p class="card-date f-11 text-lightest mb-0">
                                            {{ $reply->created_at->timezone(company()->timezone)->translatedFormat(company()->date_format . ' ' . company()->time_format) }}
                                        </p>

                                        @if ($user->id == user()->id || in_array('admin', user_roles()))
                                            <div class="dropdown ml-auto message-action">
                                                <button
                                                    class="btn btn-lg f-14 p-0 text-lightest text-capitalize rounded  dropdown-toggle"
                                                    type="button" data-toggle="dropdown" aria-haspopup="true"
                                                    aria-expanded="false">
                                                    <i class="fa fa-ellipsis-h"></i>
                                                </button>

                                                <div class="dropdown-menu dropdown-menu-right border-grey rounded b-shadow-4 p-0"
                                                    aria-labelledby="dropdownMenuLink" tabindex="0">

                                                    <a class="dropdown-item delete-message"
                                                        data-row-id="{{ $reply->id }}"
                                                        data-user-id="{{ $user->id }}"
                                                        href="javascript:;">@lang('app.delete')</a>
                                                </div>
                                            </div>
                                        @endif

                                    </div>
                                    @if ($reply->items != '')
                                        <div class="card-text text-dark-grey text-justify mb-0">
                                            <span class="ql-editor f-13 px-0">Standar Bersih untuk:
                                                <b>{{ $reply->items }}</b></span>
                                        </div>
                                    @endif

                                    @if ($reply->message != '')
                                        <div class="card-text text-dark-grey text-justify mb-2">
                                            <span class="ql-editor f-13 px-0">{!! nl2br($reply->message) !!}</span>
                                        </div>
                                    @endif


                                    <div class="d-flex flex-wrap">
                                        @foreach ($reply->files as $file)
                                            <x-file-card :fileName="$file->filename" :dateAdded="$file->created_at->diffForHumans()">
                                                @if ($file->icon == 'images')
                                                    <img src="{{ $file->file_url }}">
                                                @else
                                                    <i class="fa {{ $file->icon }} text-lightest"></i>
                                                @endif

                                                <x-slot name="action">
                                                    <div class="dropdown ml-auto file-action">
                                                        <button
                                                            class="btn btn-lg f-14 p-0 text-lightest text-capitalize rounded  dropdown-toggle"
                                                            type="button" data-toggle="dropdown" aria-haspopup="true"
                                                            aria-expanded="false">
                                                            <i class="fa fa-ellipsis-h"></i>
                                                        </button>

                                                        <div class="dropdown-menu dropdown-menu-right border-grey rounded b-shadow-4 p-0"
                                                            aria-labelledby="dropdownMenuLink" tabindex="0">

                                                            <a class="dropdown-item" target="_blank"
                                                                href="{{ $file->file_url }}">@lang('app.view')</a>

                                                            <a class="dropdown-item"
                                                                href="{{ route('schedule-files.download', md5($file->id)) }}">@lang('app.download')</a>

                                                            @if (user()->id == $user->id)
                                                                <a class="dropdown-item delete-file"
                                                                    data-row-id="{{ $file->id }}"
                                                                    href="javascript:;">@lang('app.delete')</a>
                                                            @endif
                                                        </div>
                                                    </div>
                                                </x-slot>
                                            </x-file-card>
                                        @endforeach
                                    </div>

                                </div>

                            </div>
                        </div><!-- card end -->
                    @endforeach

                </div>
                <!-- MESSAGE END -->

                <div class="col-md-12 border-top border-right d-none mb-5" id="reply-section">
                    <div class="form-group my-3">
                        <p class="f-w-500">
                            @lang('app.to'): {{ mb_ucwords($schedule->worker->name) }} (worker)
                        </p>
                        <x-forms.label fieldId="parent_label" :fieldLabel="__('Standar')" fieldName="parent_label">
                        </x-forms.label>
                        <select class="form-control select-picker mb-3" name="items" id="items"
                            data-live-search="true">
                            @foreach ($schedule->items as $item)
                                <option value="{{ $item->item_name }}">{{ $item->item_name }}</option>
                            @endforeach
                        </select>

                        <x-forms.label fieldId="parent_label" :fieldLabel="__('Description')" fieldName="parent_label">
                        </x-forms.label>
                        <div id="description"></div>
                        <textarea name="message" id="description-text" class="d-none"></textarea>
                    </div>
                    <div class="my-3">
                        <a class="f-15 f-w-500" href="javascript:;" id="add-file"><i
                                class="fa fa-paperclip font-weight-bold mr-1"></i>@lang('modules.projects.uploadFile')</a>
                    </div>
                    <x-forms.file-multiple class="mr-0 mr-lg-2 mr-md-2 upload-section d-none" fieldLabel=""
                        fieldName="file[]" fieldId="task-file-upload-dropzone" />
                </div>

                <div class="ticket-reply-back justify-content-start px-lg-4 px-md-4 px-3 py-3  d-flex bg-white border-top-grey border-right-grey"
                    id="reply-section-action">
                    <x-forms.button-primary class="reply-button mr-3" icon="reply">@lang('app.reply')
                    </x-forms.button-primary>
                    <x-forms.link-secondary :link="route('schedule-inspection.index')" icon="arrow-left">@lang('app.back')
                    </x-forms.link-secondary>
                </div>
                <div class="ticket-reply-back flex-row justify-content-start px-lg-4 px-md-4 px-3 py-3 c-inv-btns bg-white border-top-grey border-right-grey d-none"
                    id="reply-section-action-2">
                    @if ($editInspectionPermission == 'all')
                        <div class="inv-action dropup mr-3">
                            <button class="btn-primary dropdown-toggle" type="button" data-toggle="dropdown"
                                aria-haspopup="true" aria-expanded="false">
                                @lang('app.submit')<span><i class="fa fa-chevron-up f-15 text-white"></i></span>
                            </button>
                            <!-- DROPDOWN - INFORMATION -->
                            <ul class="dropdown-menu" aria-labelledby="dropdownMenuBtn" tabindex="0">
                                <li>
                                    <a class="dropdown-item f-14 text-dark submit-ticket" href="javascript:;"
                                        data-status="open">
                                        <x-status color="yellow" :value="__('Submit only')" />
                                    </a>
                                </li>
                                <li>
                                    <a class="dropdown-item f-14 text-dark submit-ticket" href="javascript:;"
                                        data-status="pending">
                                        <x-status color="red" :value="__('modules.tickets.submitPending')" />
                                    </a>
                                </li>
                                <li>
                                    <a class="dropdown-item f-14 text-dark submit-ticket" href="javascript:;"
                                        data-status="resolved">
                                        <x-status color="dark-green" :value="__('modules.tickets.submitResolved')" />
                                    </a>
                                </li>
                            </ul>
                        </div>
                    @endif
                    <x-forms.link-secondary id="cancel-reply" class="border-0" link="javascript:;">@lang('app.cancel')
                    </x-forms.link-secondary>

                </div>
            </x-form>
        </div>
        <!-- LEFT END -->

        @if ($editInspectionPermission == 'all')
            <!-- RIGHT START -->
            <div class="mobile-close-overlay w-100 h-100" id="close-tickets-overlay"></div>
            <div class="ticket-right bg-white" id="ticket-detail-contact">
                <a class="d-block d-lg-none close-it" id="close-tickets"><i class="fa fa-times"></i></a>
                <div class="border-bottom-grey">
                    <span class="border-bottom-grey">
                        <p class="f-15 mb-0 p-3">Schedule Details</p>
                    </span>
                </div>
                <div class="tab-pane fade show active" id="nav-details" role="tabpanel"
                    aria-labelledby="nav-detail-tab">
                    <x-form id="updateTicket1">
                        <div class="ticket-filters w-100 position-relative border-bottom">
                            <div class="card-horizontal bg-white-shade ticket-contact-owner p-1 rounded-0">
                                <div class="card-img mr-3">
                                    <img class="___class_+?88___" src="{{ $schedule->worker->image_url }}"
                                        alt="{{ mb_ucwords($schedule->worker->name) }}">
                                </div>
                                <div class="card-body border-0 p-0 w-100">
                                    <h4 class="card-title f-14 font-weight-normal mb-0">
                                        <a class="text-dark-grey"
                                            @if ($schedule->worker->hasRole('employee')) href="{{ route('employees.show', $schedule->worker->id) }}"
                                        @else
                                            href="{{ route('clients.show', $schedule->worker->id) }}" @endif>
                                            {{ mb_ucwords($schedule->worker->name) }} (worker)
                                        </a>
                                    </h4>
                                </div>
                            </div>
                            <div class="px-3">
                                <div class="more-filter-items">
                                    <x-forms.label class="my-3" fieldId="agent_id" :fieldLabel="__('modules.tickets.agent')">
                                    </x-forms.label>
                                    <x-forms.input-group>
                                        <select class="form-control select-picker " name="agent_id" id="agent_id"
                                            data-live-search="true" data-container="body" data-size="8">
                                            <option value="">--</option>
                                            @foreach ($groups as $group)
                                                @if (count($group->enabledAgents) > 0)
                                                    <optgroup label="{{ mb_ucwords($group->group_name) }}">
                                                        @foreach ($group->enabledAgents as $agent)
                                                            <x-user-option :user="$agent->user" :selected="$agent->user->id == $schedule->agent_id">
                                                            </x-user-option>
                                                        @endforeach
                                                    </optgroup>
                                                @endif
                                            @endforeach
                                        </select>
                                        <x-slot name="append">
                                            <button id="addAgent" type="button"
                                                class="btn btn-outline-secondary border-grey">@lang('app.add')</button>
                                        </x-slot>
                                    </x-forms.input-group>
                                </div>
                                <div class="more-filter-items">
                                    <x-forms.select fieldId="priority" :fieldLabel="__('modules.tasks.priority')"
                                        fieldName="priority" data-container="body">
                                        <option @if ($schedule->priority == 'low') selected @endif value="low">@lang('app.low')</option>
                                        <option @if ($schedule->priority == 'medium') selected @endif value="medium">@lang('app.medium')</option>
                                        <option @if ($schedule->priority == 'high') selected @endif value="high">@lang('app.high')</option>
                                        <option @if ($schedule->priority == 'urgent') selected @endif value="urgent">@lang('app.urgent')</option>
                                    </x-forms.select>
                                </div>
                                <div class="more-filter-items">
                                    <x-forms.label class="mt-3 mb-1" fieldId="tags" :fieldLabel="__('Standar Bersih')">
                                    </x-forms.label>
                                    @foreach ($schedule->items as $item)
                                        <?php $i = $loop->index; ?>
                                        <div class="form-check mb-2">
                                            <input type="hidden" name="items_name[{{ $i }}]" value=0>
                                            <input class="form-check-input" type="checkbox"
                                                name="items_name[{{ $i }}]" value="1"
                                                {{ $item->check ? 'checked' : '' }}>
                                            <input type="hidden" name="item_ids[]" value="{{ $item->id }}">
                                            <label class="form-check-label px-2 mt-1" for="navbar{{ $item->id }}">
                                                {{ mb_ucwords($item->item_name) }}
                                            </label>
                                        </div>
                                    @endforeach
                                </div>
                            </div>
                        </div>
                        <div class="ticket-update bg-white px-4 py-3">
                            <x-forms.button-primary class="ml-none d-flex submit-update fixed-bottom">
                                @lang('app.update') Status Standar Bersih
                            </x-forms.button-primary>
                        </div>
                    </x-form>
                </div>
            </div>
            <!--RIGHT END -->
        @endif
    </div>
    <!-- END -->
@endsection

@push('scripts')
    <script src="{{ asset('vendor/jquery/dropzone.min.js') }}"></script>
    <script src="{{ asset('vendor/jquery/tagify.min.js') }}"></script>

    <script>
        $(document).ready(function() {
            quillImageLoad('#description');
        });

        $('.reply-button').click(function() {
            $('#reply-section-action').toggleClass('d-none d-flex');
            $('#reply-section-action-2').toggleClass('d-none flex-row');
            $('#reply-section').removeClass('d-none');
            window.scrollTo(0, document.body.scrollHeight);
        });

        $('#cancel-reply').click(function() {
            $('#reply-section-action').toggleClass('d-none d-flex');
            $('#reply-section-action-2').toggleClass('d-none flex-row');
            $('#reply-section').addClass('d-none');
            window.scrollTo(0, document.body.scrollHeight);
        });

        $('#add-file').click(function() {
            $('.upload-section').removeClass('d-none');
            $('#add-file').addClass('d-none');
            window.scrollTo(0, document.body.scrollHeight);
        });

        var input = document.querySelector('input[name=tags]'),
            // init Tagify script on the above inputs
            tagify = new Tagify(input);

        Dropzone.autoDiscover = false;
        //Dropzone class
        taskDropzone = new Dropzone("div#task-file-upload-dropzone", {
            dictDefaultMessage: "{{ __('app.dragDrop') }}",
            url: "{{ route('schedule-files.store') }}",
            headers: {
                'X-CSRF-TOKEN': '{{ csrf_token() }}'
            },
            paramName: "file",
            maxFilesize: DROPZONE_MAX_FILESIZE,
            maxFiles: 10,
            autoProcessQueue: false,
            uploadMultiple: true,
            addRemoveLinks: true,
            parallelUploads: 10,
            acceptedFiles: DROPZONE_FILE_ALLOW,
            init: function() {
                taskDropzone = this;
            }
        });
        taskDropzone.on('sending', function(file, xhr, formData) {
            var ids = $('#schedule_reply_id').val();
            formData.append('schedule_reply_id', ids);
            formData.append('schedule_id', '{{ $schedule->id }}');
            $.easyBlockUI();
        });
        taskDropzone.on('uploadprogress', function() {
            $.easyBlockUI();
        });
        taskDropzone.on('completemultiple', function() {
            var msgs = "@lang('messages.addDiscussion')";
            window.location.href = "{{ route('schedule-inspection.show', $schedule->id) }}";
        });

        $('.submit-ticket').click(function() {
            var note = document.getElementById('description').children[0].innerHTML;
            document.getElementById('description-text').value = note;
            var status = $(this).data('status');
            $('#status').val(status);
            const url = "{{ route('schedule-inspection.update', $schedule->id) }}";

            $.easyAjax({
                url: url,
                container: '#updateTicket2',
                type: "POST",
                blockUI: true,
                data: $('#updateTicket2').serialize(),
                success: function(response) {

                    if (response.status == 'success') {
                        if (taskDropzone.getQueuedFiles().length > 0) {
                            $('#schedule_reply_id').val(response.reply_id);
                            taskDropzone.processQueue();
                        } else {
                            window.location.href =
                                "{{ route('schedule-inspection.show', $schedule->id) }}";
                        }
                    }
                }
            });
        });

        $('.submit-update').click(function() {
            $.easyAjax({
                url: "{{ route('schedule-inspection.update_other_data', $schedule->id) }}",
                container: '#updateTicket1',
                type: "POST",
                blockUI: true,
                disableButton: true,
                buttonSelector: ".submit-update",
                data: $('#updateTicket1').serialize(),
                success: function(response) {
                    if (response.status == 'success') {
                        window.location.href =
                            "{{ route('schedule-inspection.show', $schedule->id) }}";
                    }
                }
            })
        });

        $('body').on('click', '.delete-file', function() {
            var id = $(this).data('row-id');
            var replyFile = $(this);
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.recoverRecord')",
                icon: 'warning',
                showCancelButton: true,
                focusConfirm: false,
                confirmButtonText: "@lang('messages.confirmDelete')",
                cancelButtonText: "@lang('app.cancel')",
                customClass: {
                    confirmButton: 'btn btn-primary mr-3',
                    cancelButton: 'btn btn-secondary'
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    var url = "{{ route('schedule-files.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                replyFile.closest('.card').remove();
                            }
                        }
                    });
                }
            });
        });

        $('body').on('click', '.delete-ticket', function() {
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.recoverRecord')",
                icon: 'warning',
                showCancelButton: true,
                focusConfirm: false,
                confirmButtonText: "@lang('messages.confirmDelete')",
                cancelButtonText: "@lang('app.cancel')",
                customClass: {
                    confirmButton: 'btn btn-primary mr-3',
                    cancelButton: 'btn btn-secondary'
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    var url = "{{ route('schedule-inspection.destroy', $schedule->id) }}";

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                window.location.href =
                                    "{{ route('schedule-inspection.index') }}";
                            }
                        }
                    });
                }
            });
        });

        $('body').on('click', '.delete-message', function() {
            var id = $(this).data('row-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.recoverRecord')",
                icon: 'warning',
                showCancelButton: true,
                focusConfirm: false,
                confirmButtonText: "@lang('messages.confirmDelete')",
                cancelButtonText: "@lang('app.cancel')",
                customClass: {
                    confirmButton: 'btn btn-primary mr-3',
                    cancelButton: 'btn btn-secondary'
                },
                showClass: {
                    popup: 'swal2-noanimation',
                    backdrop: 'swal2-noanimation'
                },
                buttonsStyling: false
            }).then((result) => {
                if (result.isConfirmed) {
                    var url = "{{ route('schedule-replies.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                $('#message-' + id).remove();
                            }
                        }
                    });
                }
            });
        });

        function scrollToBottom(divId) {
            var myDiv = document.getElementById(divId);
            myDiv.scrollTop = myDiv.scrollHeight;
        }

        scrollToBottom('ticketMsg');
    </script>
@endpush
