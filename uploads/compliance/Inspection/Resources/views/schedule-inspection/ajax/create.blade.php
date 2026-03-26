@php
    $manageTypePermission = user()->permission('manage_ticket_type');
    $manageAgentPermission = user()->permission('manage_ticket_agent');
    $manageChannelPermission = user()->permission('manage_ticket_channel');
@endphp

<link rel="stylesheet" href="{{ asset('vendor/css/dropzone.min.css') }}">
<link rel="stylesheet" href="{{ asset('vendor/css/tagify.css') }}">

<div class="row">
    <div class="col-sm-12">
        <x-form id="save-ticket-data-form">
            <input type="hidden" id="replyID">
            <div class="add-client bg-white rounded">
                <h4 class="mb-0 p-20 f-21 font-weight-normal text-capitalize border-bottom-grey">
                    @lang('modules.tickets.ticketDetail')</h4>
                <div class="row p-20">
                    @if (!in_array('client', user_roles()))
                        @if ($addPermission == 'all')
                            @if (!(isset($client)) && !(isset($employee)))
                                <div class="col-md-4">
                                    <div class="form-group my-3">
                                        <x-forms.label fieldId="requester-client"
                                                       :fieldLabel="__('modules.tickets.requester')"/>
                                        <div class="d-flex">
                                            <x-forms.radio fieldId="requester-client" :fieldLabel="__('app.client')"
                                                           fieldName="requester_type" fieldValue="client"
                                                           checked="true">
                                            </x-forms.radio>
                                            <x-forms.radio fieldId="requester-employee" :fieldLabel="__('app.employee')"
                                                           fieldValue="employee"
                                                           fieldName="requester_type"></x-forms.radio>
                                        </div>
                                    </div>
                                </div>
                            @endif

                            <div class="col-md-4  @if (isset($employee)) d-none @endif " id="client-requester">
                                @if (isset($client) && !is_null($client))
                                    <x-forms.label fieldId="requester-client"
                                                   :fieldLabel="__('modules.tickets.requesterName')"/>
                                    <input type="hidden" name="client_id" id="client_id" value="{{ $client->id }}">
                                    <input type="hidden" name="requester_type" id="requester_type" value="client">
                                    <input type="text" value="{{ ucfirst($client->name) }}"
                                           class="form-control height-35 f-15 readonly-background" readonly>
                                @else
                                    <x-forms.select fieldId="client_id"
                                                    :fieldLabel="__('modules.tickets.requesterName')"
                                                    fieldName="client_id"
                                                    search="true" alignRight="true" fieldRequired="true">
                                        <option value="">--</option>
                                        @foreach ($clients as $client)
                                            <x-user-option :user="$client"/>
                                        @endforeach
                                    </x-forms.select>
                                @endif
                            </div>

                            <div class="col-md-4 @if (!(isset($employee))) d-none @endif" id="employee-requester">
                                @if(isset($employee) && !is_null($employee))
                                    <x-forms.label class="my-3" fieldId="requestuser_id"
                                                   :fieldLabel="__('modules.tickets.requesterName')"
                                                   fieldRequired="true">
                                    </x-forms.label>
                                    <input type="hidden" name="user_id" id="user_id" value="{{ $employee->id }}">
                                    <input type="hidden" name="requester_type" id="requester_type" value="employee">
                                    <input type="text" value="{{ $employee->name }}"
                                           class="form-control height-35 f-15 readonly-background" readonly>
                                @else
                                    <x-forms.label class="my-3" fieldId="requestuser_id"
                                                   :fieldLabel="__('modules.tickets.requesterName')"
                                                   fieldRequired="true">
                                    </x-forms.label>
                                    <x-forms.input-group>
                                        <select class="form-control select-picker" name="user_id" id="user_id"
                                                data-live-search="true" data-size="8">
                                            <option value="">--</option>
                                            @foreach ($employees as $employee)
                                                <x-user-option :user="$employee"/>
                                            @endforeach
                                        </select>
                                    </x-forms.input-group>
                                @endif
                            </div>
                        @else
                            <input type="hidden" name="requester_type" value="employee">
                            <input type="hidden" name="user_id" value="{{ user()->id }}">
                        @endif
                    @else
                        <input type="hidden" name="requester_type" value="client">
                        <input type="hidden" name="client_id" value="{{ user()->id }}">
                    @endif


                    <div class="col-md-12">
                        <x-forms.text :fieldLabel="__('modules.tickets.ticketSubject')" fieldName="subject"
                                      fieldRequired="true" fieldId="subject"/>
                    </div>

                    <div class="col-md-12">
                        <div class="form-group my-3">
                            <x-forms.label fieldId="description" :fieldLabel="__('app.description')"
                                           fieldRequired="true">
                            </x-forms.label>
                            <div id="description"></div>
                            <textarea name="description" id="description-text" class="d-none"></textarea>
                        </div>
                        <div class="my-3">
                            <a class="f-15 f-w-500" href="javascript:;" id="add-file"><i
                                    class="fa fa-paperclip font-weight-bold mr-1"></i>@lang('modules.projects.uploadFile')
                            </a>
                        </div>
                    </div>
                </div>

                <div class="row px-4">
                    <div class="col-md-12">
                        <x-forms.file-multiple class="mr-0 mr-lg-2 mr-md-2 upload-section d-none"
                                               :fieldLabel="__('app.add') . ' ' . __('app.file')"
                                               fieldName="file" fieldId="task-file-upload-dropzone"/>
                        <input type="hidden" name="image_url" id="image_url">
                    </div>

                </div>

                <h4 class="mb-0 p-20 f-21 font-weight-normal text-capitalize border-top-grey">
                    <a href="javascript:;" class="text-dark toggle-other-details"><i class="fa fa-chevron-down"></i>
                        @lang('modules.client.clientOtherDetails')</a>
                </h4>

                <div class="row p-20 d-none" id="other-details">

                    @if (!in_array('client', user_roles()))
                        <div class="col-md-6 col-lg-3">
                            <x-forms.label class="mt-3" fieldId="agent_id" :fieldLabel="__('modules.tickets.agent')">
                            </x-forms.label>
                            <x-forms.input-group>
                                <select class="form-control select-picker" name="agent_id" id="agent_id"
                                        data-live-search="true" data-size="8">
                                    <option value="">--</option>
                                    @foreach ($groups as $group)
                                        @if (count($group->enabledAgents) > 0)
                                            <optgroup label="{{ mb_ucwords($group->group_name) }}">
                                                @foreach ($group->enabledAgents as $agent)

                                                    <x-user-option :user="$agent->user" :agent="true"></x-user-option>

                                                @endforeach
                                            </optgroup>
                                        @endif
                                    @endforeach
                                </select>
                                @if ($manageAgentPermission == 'all')
                                    <x-slot name="append">
                                        <button id="add-agent" type="button"
                                                class="btn btn-outline-secondary border-grey"
                                                data-toggle="tooltip" data-original-title="{{ __('app.addNew').' '.__('modules.tickets.agent') }}">@lang('app.add')</button>
                                    </x-slot>
                                @endif
                            </x-forms.input-group>
                        </div>

                    @endif

                    <div class="col-md-6 col-lg-3">
                        <x-forms.select fieldId="priority" :fieldLabel="__('modules.tasks.priority')"
                                        fieldName="priority">
                            <option value="low">@lang('app.low')</option>
                            <option value="medium">@lang('app.medium')</option>
                            <option value="high">@lang('app.high')</option>
                            <option value="urgent">@lang('app.urgent')</option>
                        </x-forms.select>
                    </div>

                    <div class="col-md-6 col-lg-3">
                        <x-forms.label class="my-3" fieldId="ticket_type_id" :fieldLabel="__('modules.invoices.type')">
                        </x-forms.label>
                        <x-forms.input-group>
                            <select class="form-control select-picker" name="type_id" id="ticket_type_id"
                                    data-live-search="true" data-size="8">
                                <option value="">--</option>
                                @foreach ($types as $type)
                                    <option value="{{ $type->id }}">{{ mb_ucwords($type->type) }}</option>
                                @endforeach
                            </select>
                            @if ($manageTypePermission == 'all')
                                <x-slot name="append">
                                    <button id="add-type" type="button"
                                            class="btn btn-outline-secondary border-grey"
                                            data-toggle="tooltip" data-original-title="{{ __('app.addNew').' '.__('modules.tickets.ticketType') }}">@lang('app.add')</button>
                                </x-slot>
                            @endif
                        </x-forms.input-group>
                    </div>

                    <div class="col-md-6 col-lg-3">
                        <x-forms.label class="my-3" fieldId="ticket_channel_id"
                                       :fieldLabel="__('modules.tickets.channelName')">
                        </x-forms.label>
                        <x-forms.input-group>
                            <select class="form-control select-picker" name="channel_id" id="ticket_channel_id"
                                    data-live-search="true" data-size="8">
                                <option value="">--</option>
                                @foreach ($channels as $channel)
                                    <option value="{{ $channel->id }}">{{ mb_ucwords($channel->channel_name) }}
                                    </option>
                                @endforeach
                            </select>
                            @if ($manageChannelPermission == 'all')
                                <x-slot name="append">
                                    <button id="add-channel" type="button"
                                            class="btn btn-outline-secondary border-grey"
                                            data-toggle="tooltip" data-original-title="{{ __('app.add').' '.__('modules.tickets.ticketChannel') }}">@lang('app.add')</button>
                                </x-slot>
                            @endif
                        </x-forms.input-group>
                    </div>

                    <div class="col-md-12">
                        <x-forms.text fieldId="tags" :fieldLabel="__('modules.tickets.tags')" fieldName="tags"/>
                    </div>

                    <x-forms.custom-field :fields="$fields" class="col-md-12"></x-forms.custom-field>

                </div>

                <x-form-actions>
                    <x-forms.button-primary id="save-ticket-form" class="mr-3" icon="check">@lang('app.save')
                    </x-forms.button-primary>
                    <x-forms.button-cancel :link="route('tickets.index')" class="border-0">@lang('app.cancel')
                    </x-forms.button-cancel>
                </x-form-actions>

            </div>
        </x-form>

    </div>
</div>


<script src="{{ asset('vendor/jquery/dropzone.min.js') }}"></script>
<script src="{{ asset('vendor/jquery/tagify.min.js') }}"></script>
<script>
    $(document).ready(function () {

        $('#add-file').click(function () {
            $('.upload-section').removeClass('d-none');
            $('#add-file').addClass('d-none');
            window.scrollTo(0, document.body.scrollHeight);
        });

        Dropzone.autoDiscover = false;
        //Dropzone class
        var taskDropzone = new Dropzone("div#task-file-upload-dropzone", {
            dictDefaultMessage: "{{ __('app.dragDrop') }}",
            url: "{{ route('ticket-files.store') }}",
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
            init: function () {
                taskDropzone = this;
            }
        });
        taskDropzone.on('sending', function (file, xhr, formData) {
            var ids = $('#replyID').val();
            formData.append('ticket_reply_id', ids);
            $.easyBlockUI();
        });
        taskDropzone.on('uploadprogress', function () {
            $.easyBlockUI();
        });
        taskDropzone.on('completemultiple', function () {
            var msgs = "@lang('messages.addDiscussion')";
            window.location.href = "{{ route('tickets.index') }}";
        });

        var input = document.querySelector('input[name=tags]'),
            // init Tagify script on the above inputs
            tagify = new Tagify(input);

        quillImageLoad('#description');

        $("input[name=requester_type]").click(function () {
            $('#client-requester, #employee-requester').toggleClass('d-none');
        });

        /* open add agent modal */
        $('body').on('click', '#add-agent', function () {
            var url = "{{ route('ticket-agents.create') }}";
            $(MODAL_XL + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_XL, url);
        });

        /* open add agent modal */
        $('body').on('click', '#add-channel', function () {
            var url = "{{ route('ticketChannels.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        /* open add agent modal */
        $('body').on('click', '#add-type', function () {
            var url = "{{ route('ticketTypes.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        $('#save-ticket-form').click(function () {
            var note = document.getElementById('description').children[0].innerHTML;
            document.getElementById('description-text').value = note;

            const url = "{{ route('tickets.store') }}";

            $.easyAjax({
                url: url,
                container: '#save-ticket-data-form',
                type: "POST",
                disableButton: true,
                blockUI: true,
                buttonSelector: "#save-ticket-form",
                data: $('#save-ticket-data-form').serialize(),
                success: function (response) {
                    if (response.status == 'success') {
                        if (taskDropzone.getQueuedFiles().length > 0) {
                            $('#replyID').val(response.replyID);
                            taskDropzone.processQueue();
                        } else {
                            window.location.href = response.redirectUrl;
                        }
                    }

                }
            });
        });

        $('#create_task_category').click(function () {
            const url = "{{ route('taskCategory.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        $('#department-setting').click(function () {
            const url = "{{ route('departments.create') }}";
            $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_LG, url);
        });

        $('#client_view_task').change(function () {
            $('#clientNotification').toggleClass('d-none');
        });

        $('#set_time_estimate').change(function () {
            $('#set-time-estimate-fields').toggleClass('d-none');
        });

        $('#repeat-task').change(function () {
            $('#repeat-fields').toggleClass('d-none');
        });

        $('#dependent-task').change(function () {
            $('#dependent-fields').toggleClass('d-none');
        });

        $('.toggle-other-details').click(function () {
            $(this).find('svg').toggleClass('fa-chevron-down fa-chevron-up');
            $('#other-details').toggleClass('d-none');
        });

        $('#createTaskLabel').click(function () {
            const url = "{{ route('task-label.create') }}";
            $(MODAL_XL + ' ' + MODAL_HEADING).html('...');
            $.ajaxModal(MODAL_XL, url);
        });

        $('#add-project').click(function () {
            $(MODAL_XL).modal('show');

            const url = "{{ route('projects.create') }}";

            $.easyAjax({
                url: url,
                blockUI: true,
                container: MODAL_XL,
                success: function (response) {
                    if (response.status == "success") {
                        $(MODAL_XL + ' .modal-body').html(response.html);
                        $(MODAL_XL + ' .modal-title').html(response.title);
                        init(MODAL_XL);
                    }
                }
            });
        });

        $('#add-employee').click(function () {
            $(MODAL_XL).modal('show');

            const url = "{{ route('employees.create') }}";

            $.easyAjax({
                url: url,
                blockUI: true,
                container: MODAL_XL,
                success: function (response) {
                    if (response.status == "success") {
                        $(MODAL_XL + ' .modal-body').html(response.html);
                        $(MODAL_XL + ' .modal-title').html(response.title);
                        init(MODAL_XL);
                    }
                }
            });
        });

        init(RIGHT_MODAL);
    });
</script>
