@extends('layouts.app')

@section('content')

    <!-- SETTINGS START -->
    <div class="w-100 d-flex ">

        <x-setting-sidebar :activeMenu="$activeSettingMenu" />

        <x-setting-card>
            <x-slot name="header">
                <div class="s-b-n-header" id="tabs">
                    <nav class="tabs px-4 border-bottom-grey">
                        <div class="nav" id="nav-tab" role="tablist">

                            <a class="nav-item nav-link f-15 active agent" href="{{ route('complaint-settings.index') }}"
                                role="tab" aria-controls="nav-ticketAgents"
                                aria-selected="true">@lang('complaint::modules.complaintAgents')
                            </a>

                            <a class="nav-item nav-link f-15 type" href="{{ route('complaint-settings.index') }}?tab=type"
                                role="tab" aria-controls="nav-ticketTypes"
                                aria-selected="true">@lang('complaint::modules.complaintTypes')
                            </a>

                            <a class="nav-item nav-link f-15 channel"
                                href="{{ route('complaint-settings.index') }}?tab=channel" role="tab"
                                aria-controls="nav-ticketChannel" aria-selected="true">@lang('complaint::modules.complaintChannels')
                            </a>

                            <a class="nav-item nav-link f-15 reply-template"
                                href="{{ route('complaint-settings.index') }}?tab=reply-template" role="tab"
                                aria-controls="nav-replyTemplates" aria-selected="true">@lang('complaint::modules.complaintTemplates')
                            </a>

                            {{-- <a class="nav-item nav-link f-15 email-sync"
                                href="{{ route('complaint-settings.index') }}?tab=email-sync" role="tab"
                                aria-controls="nav-emailSync" aria-selected="true">@lang('app.menu.emailSync')
                            </a> --}}

                        </div>
                    </nav>
                </div>
            </x-slot>

            <x-slot name="buttons">
                <div class="row">

                    <div class="col-md-12 mb-2">
                        <x-forms.button-primary icon="plus" id="addTicketType" class="type-btn mb-2 d-none actionBtn">
                            @lang('app.addNew') @lang('complaint::modules.complaintTypes')
                        </x-forms.button-primary>

                        <x-forms.button-primary icon="plus" id="addAgent" class="agent-btn mb-2 d-none actionBtn">
                            @lang('app.addNew')
                            @lang('complaint::modules.complaintAgents')
                        </x-forms.button-primary>

                        <x-forms.button-primary icon="plus" id="addChannel" class="channel-btn mb-2 d-none actionBtn">
                            @lang('app.addNew')
                            @lang('complaint::modules.complaintChannels')
                        </x-forms.button-primary>

                        <x-forms.button-primary icon="plus" id="addReplyTemplate"
                            class="reply-template-btn mb-2 d-none actionBtn">
                            @lang('app.addNew') @lang('complaint::modules.complaintTemplates')
                        </x-forms.button-primary>
                    </div>

                </div>
            </x-slot>

            {{-- include tabs here --}}
            @include($view)

        </x-setting-card>

    </div>
    <!-- SETTINGS END -->

@endsection

@push('scripts')
    <script>
        /* manage menu active class */
        $('.nav-item').removeClass('active');
        const activeTab = "{{ $activeTab }}";
        $('.' + activeTab).addClass('active');

        showBtn(activeTab);

        function showBtn(activeTab) {
            $('.actionBtn').addClass('d-none');
            $('.' + activeTab + '-btn').removeClass('d-none');
        }

       $("body").on("click", "#editSettings .nav a", function(event) {
            event.preventDefault();

            $('.nav-item').removeClass('active');
            $(this).addClass('active');

            const requestUrl = this.href;

            $.easyAjax({
                url: requestUrl,
                blockUI: true,
                container: "#nav-tabContent",
                historyPush: true,
                success: function(response) {
                    if (response.status == "success") {
                        showBtn(response.activeTab);

                        $('#nav-tabContent').html(response.html);
                        init('#nav-tabContent');
                    }
                }
            });
        });

        /* delete agent */
        $('body').on('click', '.delete-agents', function() {
            var id = $(this).data('agent-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.removeAgentText')",
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
                    var url = "{{ route('complaint-agents.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        blockUI: true,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                $('.row' + id).fadeOut(100);
                            }
                        }
                    });
                }
            });
        });

        /* change agent status */
        $('body').on('change', '.change-agent-status', function() {

            var agentId = $(this).data('agent-id');
            var status = $(this).val();

            var token = '{{ csrf_token() }}';
            var url = "{{ route('complaint-agents.update', ':id') }}";
            url = url.replace(':id', agentId);

            if (typeof agentId !== 'undefined') {
                $.easyAjax({
                    type: 'PUT',
                    url: url,
                    blockUI: true,
                    data: {
                        '_token': token,
                        'status': status
                    }
                });
            }
        });


        $('body').on('click', '.delete-type', function() {
            var id = $(this).data('type-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.removeTicketText')",
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
                    var url = "{{ route('complaintTypes.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        blockUI: true,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                $('.row' + id).fadeOut(100);
                            }
                        }
                    });
                }
            });
        });

        $('body').on('click', '.edit-type', function() {
            var typeId = $(this).data('type-id');
            var url = "{{ route('complaintTypes.edit', ':id') }}";
            url = url.replace(':id', typeId);

            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        $('body').on('click', '.delete-channel', function() {
            var id = $(this).data('channel-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.removeChannelText')",
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
                    var url = "{{ route('complaintChannels.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        blockUI: true,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                $('.row' + id).fadeOut(100);
                            }
                        }
                    });
                }
            });
        });

        $('body').on('click', '.edit-channel', function() {
            var typeId = $(this).data('channel-id');
            var url = "{{ route('complaintChannels.edit', ':id') }}";
            url = url.replace(':id', typeId);

            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* open edit reply template modal */
        $('body').on('click', '.edit-template', function() {
            var templateId = $(this).data('template-id');
            var url = "{{ route('complaintTemplates.edit', ':id') }}";
            url = url.replace(':id', templateId);

            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* delete template */
        $('body').on('click', '.delete-template', function() {
            var id = $(this).data('template-id');
            Swal.fire({
                title: "@lang('messages.sweetAlertTitle')",
                text: "@lang('messages.removeTemplateText')",
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
                    var url = "{{ route('complaintTemplates.destroy', ':id') }}";
                    url = url.replace(':id', id);

                    var token = "{{ csrf_token() }}";

                    $.easyAjax({
                        type: 'POST',
                        url: url,
                        blockUI: true,
                        data: {
                            '_token': token,
                            '_method': 'DELETE'
                        },
                        success: function(response) {
                            if (response.status == "success") {
                                $('.row' + id).fadeOut(100);
                            }
                        }
                    });
                }
            });
        });

        /* open add agent modal */
        $('body').on('click', '#addAgent', function() {
            var url = "{{ route('complaint-agents.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* open add agent modal */
        $('body').on('click', '#addChannel', function() {
            var url = "{{ route('complaintChannels.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* open creat reply template modal */
        $('body').on('click', '#addReplyTemplate', function() {
            var url = "{{ route('complaintTemplates.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* open add agent modal */
        $('body').on('click', '#addTicketType', function() {
            var url = "{{ route('complaintTypes.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* change agent group */
        $('.change-agent-group').change(function() {

            var agentId = $(this).data('agent-id');
            var groupId = $(this).val();
            var token = '{{ csrf_token() }}';
            var url = "{{ route('complaint_agents.update_group', ':id') }}";
            url = url.replace(':id', agentId);

            $.easyAjax({
                type: 'POST',
                url: url,
                blockUI: true,
                data: {
                    '_token': token,
                    'groupId': groupId
                }
            });
        });

    </script>
@endpush
