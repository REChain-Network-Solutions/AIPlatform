@extends('layouts.app')

@push('datatable-styles')
    @include('sections.datatable_css')
@endpush

@section('filter-section')

    <x-filters.filter-box>
        <!-- DATE START -->
        <div class="select-box d-flex pr-2 border-right-grey border-right-grey-sm-0">
            <p class="mb-0 pr-2 f-14 text-dark-grey d-flex align-items-center">@lang('app.duration')</p>
            <div class="select-status d-flex">
                <input type="text"
                    class="position-relative text-dark form-control border-0 p-2 text-left f-14 f-w-500 border-additional-grey"
                    id="datatableRange" placeholder="@lang('placeholders.dateRange')">
            </div>
        </div>
        <!-- DATE END -->

        <!-- STATUS START -->
        <div class="select-box d-flex py-2 px-lg-2 px-md-2 px-0 border-right-grey border-right-grey-sm-0">
            <p class="mb-0 pr-2 f-14 text-dark-grey d-flex align-items-center">@lang('app.status')</p>
            <div class="select-status">
                <select class="form-control select-picker" name="status" id="ticket-status">
                    <option {{ request('status') == 'all' ? 'selected' : '' }} value="all">@lang('app.all')</option>
                    <option {{ request('status') == 'open' || request('status') == '' ? 'selected' : '' }}
                        value="open">
                        @lang('complaint::modules.totalOpen')</option>
                    <option {{ request('status') == 'pending' ? 'selected' : '' }} value="pending">
                        @lang('complaint::modules.totalPending')</option>
                    <option {{ request('status') == 'resolved' ? 'selected' : '' }} value="resolved">
                        @lang('complaint::modules.totalResolved')</option>
                    <option {{ request('status') == 'closed' ? 'selected' : '' }} value="closed">
                        @lang('complaint::modules.totalClosed')</option>
                </select>
            </div>
        </div>
        <!-- STATUS END -->

        <!-- SEARCH BY TASK START -->
        <div class="task-search d-flex  py-1 px-lg-3 px-0 border-right-grey align-items-center">
            <form class="w-100 mr-1 mr-lg-0 mr-md-1 ml-md-1 ml-0 ml-lg-0">
                <div class="input-group bg-grey rounded">
                    <div class="input-group-prepend">
                        <span class="input-group-text border-0 bg-additional-grey">
                            <i class="fa fa-search f-13 text-dark-grey"></i>
                        </span>
                    </div>
                    <input type="text" class="form-control f-14 p-1 border-additional-grey" id="search-text-field"
                        placeholder="@lang('app.startTyping')">
                </div>
            </form>
        </div>
        <!-- SEARCH BY TASK END -->

        <!-- RESET START -->
        <div class="select-box d-flex py-1 px-lg-2 px-md-2 px-0">
            <x-forms.button-secondary class="btn-xs d-none" id="reset-filters" icon="times-circle">
                @lang('app.clearFilters')
            </x-forms.button-secondary>
        </div>
        <!-- RESET END -->

        <!-- MORE FILTERS START -->
        <x-filters.more-filter-box>
            <!-- AGENT START -->
            @if (!in_array('client', user_roles()))
                <div class="more-filter-items">
                    <label class="f-14 text-dark-grey mb-12 text-capitalize" for="usr">@lang('complaint::modules.complaintAgents')</label>
                    <div class="select-filter mb-4">
                        <div class="select-others">
                            <select class="form-control select-picker" name="agent_id" id="agent_id"
                                data-live-search="true" data-container="body" data-size="8">
                                <option value="all">@lang('app.all')</option>
                                @if ($groups)
                                    @foreach ($groups as $group)
                                        <optgroup label="{{ mb_ucwords($group->group_name) }}">
                                            @foreach ($group->enabledAgents as $agent)
                                                <x-user-option :user="$agent->user" :selected="request('agent') == $agent->user->id ||
                                                    (request('agent') == 'me' && $agent->user->id == user()->id)" />
                                            @endforeach
                                        </optgroup>
                                    @endforeach
                                @endif
                            </select>
                        </div>
                    </div>
                </div>
            @endif
            <!-- AGENT END -->


            <div class="more-filter-items">
                <label class="f-14 text-dark-grey mb-12 text-capitalize" for="usr">@lang('modules.tasks.priority')</label>
                <div class="select-filter mb-4">
                    <div class="select-others">
                        <select class="form-control select-picker" name="priority" id="priority" data-live-search="true"
                            data-container="body" data-size="8">
                            <option value="all">@lang('app.all')</option>
                            <option value="low">@lang('modules.tasks.low')</option>
                            <option value="medium">@lang('modules.tasks.medium')</option>
                            <option value="high">@lang('modules.tasks.high')</option>
                            <option value="urgent">@lang('complaint::modules.urgent')</option>
                        </select>
                    </div>
                </div>
            </div>
            <div class="more-filter-items">
                <label class="f-14 text-dark-grey mb-12 text-capitalize" for="usr">@lang('complaint::modules.complaintChannels')</label>
                <div class="select-filter mb-4">
                    <div class="select-others">
                        <select class="form-control select-picker" id="channel_id" data-container="body">
                            <option value="all">@lang('app.all')</option>
                            @foreach ($channels as $channel)
                                <option value="{{ $channel->id }}">{{ mb_ucwords($channel->channel_name) }}</option>
                            @endforeach
                        </select>
                    </div>
                </div>
            </div>
            <div class="more-filter-items">
                <label class="f-14 text-dark-grey mb-12 text-capitalize" for="usr">@lang('modules.invoices.type')</label>
                <div class="select-filter mb-4">
                    <div class="select-others">
                        <select class="form-control select-picker" id="type_id" data-live-search="true" data-size="8"
                            data-container="body">
                            <option value="all">@lang('app.all')</option>
                            @foreach ($types as $type)
                                <option value="{{ $type->id }}">{{ mb_ucwords($type->type) }}</option>
                            @endforeach
                        </select>
                    </div>
                </div>
            </div>

            <div class="more-filter-items">
                <label class="f-14 text-dark-grey mb-12 text-capitalize" for="usr">@lang('complaint::modules.complaintTags')</label>
                <div class="select-filter mb-4">
                    <div class="select-others">
                        <select class="form-control select-picker" id="tag_id" data-live-search="true" data-size="8"
                            data-container="body">
                            <option value="all">@lang('app.all')</option>
                            @foreach ($tags as $tag)
                                <option value="{{ $tag->id }}">{{ mb_ucwords($tag->tag_name) }}</option>
                            @endforeach
                        </select>
                    </div>
                </div>
            </div>

        </x-filters.more-filter-box>
        <!-- MORE FILTERS END -->
    </x-filters.filter-box>

@endsection

@php
    $addTicketPermission = user()->permission('add_tickets');
@endphp

@section('content')
    <!-- CONTENT WRAPPER START -->
    <div class="content-wrapper">
        <div class="row row-cols-lg-5 my-3">

            <div class="col mb-4">
                <a href="javascript:;" data-status="all" class="widget-filter-status">
                    <x-cards.widget :title="__('complaint::modules.total')" value="0" icon="ticket-alt" widgetId="totalComplaints" />
                </a>
            </div>

            <div class="col mb-4">
                <a href="javascript:;" data-status="closed" class="widget-filter-status">
                    <x-cards.widget :title="__('complaint::modules.totalClosed')" value="0" icon="ticket-alt" widgetId="closedComplaints" />
                </a>
            </div>

            <div class="col mb-4">
                <a href="javascript:;" data-status="open" class="widget-filter-status">
                    <x-cards.widget :title="__('complaint::modules.totalOpen')" value="0" icon="ticket-alt" widgetId="openComplaints" />
                </a>
            </div>

            <div class="col mb-4">
                <a href="javascript:;" data-status="pending" class="widget-filter-status">
                    <x-cards.widget :title="__('complaint::modules.totalPending')" value="0" icon="ticket-alt" widgetId="pendingComplaints" />
                </a>
            </div>

            <div class="col">
                <a href="javascript:;" data-status="resolved" class="widget-filter-status">
                    <x-cards.widget :title="__('complaint::modules.totalResolved')" value="0" icon="ticket-alt" widgetId="resolvedComplaints" />
                </a>
            </div>

        </div>

        <!-- Add Task Export Buttons Start -->
        <div class="d-flex justify-content-between action-bar">
            <div id="table-actions" class="flex-grow-1 align-items-center mt-3">
                <x-forms.link-primary :link="route('complaint.create')" class="mr-3 openRightModal float-left" icon="plus">
                    @lang('app.addNew') @lang('complaint::modules.complaint')
                </x-forms.link-primary>

                {{-- @if (in_array('admin', user_roles()))
                    <x-forms.button-secondary icon="pencil-alt" class="mr-3 float-left" id="add-ticket">
                        @lang('modules.ticketForm')
                    </x-forms.button-secondary>
                @endif --}}
            </div>
        </div>

        <!-- Add Task Export Buttons End -->
        <!-- Task Box Start -->
        <div class="d-flex flex-column w-tables rounded mt-3 bg-white">

            {!! $dataTable->table(['class' => 'table table-hover border-0 w-100']) !!}

        </div>
        <!-- Task Box End -->
    </div>
    <!-- CONTENT WRAPPER END -->
@endsection

@push('scripts')
    @include('sections.datatable_js')

    <script>
        var ticketFilterStatus = "{{ request('dashboard-ticket-status') }}";

        $('#ticket-table').on('preXhr.dt', function(e, settings, data) {

            var dateRangePicker = $('#datatableRange').data('daterangepicker');
            var startDate = $('#datatableRange').val();

            if (startDate == '') {
                startDate = null;
                endDate = null;
            } else {
                startDate = dateRangePicker.startDate.format('{{ company()->moment_date_format }}');
                endDate = dateRangePicker.endDate.format('{{ company()->moment_date_format }}');
            }

            @if (request('startDate') != '' && request('endDate') != '')
                startDate = '{{ request('startDate') }}';
                endDate = '{{ request('endDate') }}';
            @endif

            var agentId = $('#agent_id').val();
            if (agentId == "") {
                agentId = 0;
            }

            var status = $('#ticket-status').val();
            if (status == "") {
                status = 0;
            }

            var priority = $('#priority').val();
            if (priority == "") {
                priority = 0;
            }

            var channelId = $('#channel_id').val();
            if (channelId == "") {
                channelId = 0;
            }

            var typeId = $('#type_id').val();
            if (typeId == "") {
                typeId = 0;
            }

            var tagId = $('#tag_id').val();
            if (tagId == "") {
                tagId = 0;
            }
            var searchText = $('#search-text-field').val();

            data['startDate'] = startDate;
            data['endDate'] = endDate;
            data['agentId'] = agentId;
            data['priority'] = priority;
            data['channelId'] = channelId;
            data['typeId'] = typeId;
            data['tagId'] = tagId;
            data['ticketStatus'] = status;
            data['searchText'] = searchText;
            if (ticketFilterStatus != '') {
                data['ticketFilterStatus'] = ticketFilterStatus;
            }


        });
        const showTable = () => {
            window.LaravelDataTables["ticket-table"].draw(false);
            refreshCount();
        }

        $('body').on('click', '.delete-table-row', function() {
            var id = $(this).data('ticket-id');
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
                    var url = "{{ route('complaint.destroy', ':id') }}";
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
                                showTable();
                            }
                        }
                    });
                }
            });
        });

        $('#ticket-table').on('change', '.change-status', function() {
            var url = "{{ route('complaint.change-status') }}";
            var token = "{{ csrf_token() }}";
            var id = $(this).data('ticket-id');
            var status = $(this).val();

            if (id != "" && status != "") {
                $.easyAjax({
                    url: url,
                    type: "POST",
                    container: '.content-wrapper',
                    blockUI: true,
                    data: {
                        '_token': token,
                        ticketId: id,
                        status: status,
                    },
                    success: function(data) {
                        if (data.status == 'success') {
                            refreshCount();
                        }
                        window.LaravelDataTables["ticket-table"].draw(false);
                    }
                });

            }
        });

        $('#agent_id, #ticket-status, #priority, #channel_id, #type_id, #tag_id')
            .on('change keyup',
                function() {
                    if ($('#ticket-status').val() != "not finished") {
                        $('#reset-filters').removeClass('d-none');
                        showTable();
                    } else if ($('#agent_id').val() != "all") {
                        $('#reset-filters').removeClass('d-none');
                        showTable();
                    } else if ($('#priority').val() != "all") {
                        $('#reset-filters').removeClass('d-none');
                        showTable();
                    } else if ($('#channel_id').val() != "all") {
                        $('#reset-filters').removeClass('d-none');
                        showTable();
                    } else if ($('#type_id').val() != "all") {
                        $('#reset-filters').removeClass('d-none');
                        showTable();
                    } else if ($('#tag_id').val() != "all") {
                        $('#reset-filters').removeClass('d-none');
                        showTable();
                    } else {
                        $('#reset-filters').addClass('d-none');
                        showTable();
                    }
                });

        $('#search-text-field').on('keyup', function() {
            if ($('#search-text-field').val() != "") {
                $('#reset-filters').removeClass('d-none');
                showTable();
            }
        });

        $('.widget-filter-status').click(function() {
            var status = $(this).data('status');
            $('#ticket-status').val(status);
            $('#ticket-status').selectpicker('refresh');
            ticketFilterStatus = '';
            showTable();
        });

        $('#reset-filters,#reset-filters-2').click(function() {
            $('#filter-form')[0].reset();
            $('.filter-box .select-picker').selectpicker("refresh");
            $('#reset-filters').addClass('d-none');
            showTable();
        });

        function refreshCount() {
            var dateRangePicker = $('#datatableRange').data('daterangepicker');
            var startDate = $('#datatableRange').val();

            if (startDate == '') {
                startDate = null;
                endDate = null;
            } else {
                startDate = dateRangePicker.startDate.format('{{ company()->moment_date_format }}');
                endDate = dateRangePicker.endDate.format('{{ company()->moment_date_format }}');
            }

            @if (!is_null(request('status')) && !is_null(request('startDate')) && !is_null(request('endDate')))
                startDate = '{{ request('startDate') }}';
                endDate = '{{ request('endDate') }}';
            @endif

            if (endDate == '') {
                endDate = null;
            }
            var agentId = $('#agent_id').val();
            if (agentId == "") {
                agentId = 0;
            }

            var status = $('#ticket-status').val();
            if (status == "") {
                status = 0;
            }


            var priority = $('#priority').val();
            if (priority == "") {
                priority = 0;
            }

            var channelId = $('#channel_id').val();
            if (channelId == "") {
                channelId = 0;
            }

            var typeId = $('#type_id').val();
            if (typeId == "") {
                typeId = 0;
            }

            var url = "{{ route('complaint.refresh_count') }}";
            $.easyAjax({
                type: 'POST',
                url: url,
                data: {
                    'startDate': startDate,
                    'endDate': endDate,
                    'agentId': agentId,
                    'ticketStatus': status,
                    'priority': priority,
                    'channelId': channelId,
                    'typeId': typeId,
                    '_token': '{{ csrf_token() }}'
                },
                success: function(response) {
                    $('#totalComplaints').html(response.totalComplaints);
                    $('#closedComplaints').html(response.closedComplaints);
                    $('#openComplaints').html(response.openComplaints);
                    $('#pendingComplaints').html(response.pendingComplaints);
                    $('#resolvedComplaints').html(response.resolvedComplaints);
                    console.log(response);
                }
            });
        }

        $('body').on('click', '#add-ticket', function() {
            window.location.href = "{{ route('complaint-form.index') }}";
        });

        $(document).ready(function() {
            @if (!is_null(request('startDate')) && !is_null(request('endDate')))
                $('#datatableRange').val('{{ request('startDate') }}' +
                    ' @lang('app.to') ' + '{{ request('endDate') }}');
                $('#datatableRange').data('daterangepicker').setStartDate("{{ request('startDate') }}");
                $('#datatableRange').data('daterangepicker').setEndDate("{{ request('endDate') }}");
                refreshCount();
            @else
                refreshCount();
            @endif
        });
    </script>
@endpush
