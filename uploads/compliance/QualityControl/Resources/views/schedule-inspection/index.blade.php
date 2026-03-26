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
                <input type="text" class="position-relative text-dark form-control border-0 p-2 text-left f-14 f-w-500 border-additional-grey"
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
                    <option {{ (request('status') == 'open' || request('status') == '' ) ? 'selected' : '' }} value="open">
                        @lang('modules.tickets.totalOpenTickets')</option>
                    <option {{ request('status') == 'pending' ? 'selected' : '' }} value="pending">
                        @lang('modules.tickets.totalPendingTickets')</option>
                    <option {{ request('status') == 'resolved' ? 'selected' : '' }} value="resolved">
                        @lang('modules.tickets.totalResolvedTickets')</option>
                    <option {{ request('status') == 'closed' ? 'selected' : '' }} value="closed">
                        @lang('modules.tickets.totalClosedTickets')</option>
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
                    <label class="f-14 text-dark-grey mb-12 text-capitalize"
                        for="usr">@lang('modules.tickets.agent')</label>
                    <div class="select-filter mb-4">
                        <div class="select-others">
                            <select class="form-control select-picker" name="agent_id" id="agent_id" data-live-search="true"
                                data-container="body" data-size="8">
                                <option value="all">@lang('app.all')</option>
                                @if ($groups)

                                    @foreach ($groups as $group)
                                        <optgroup label="{{ mb_ucwords($group->group_name) }}">
                                            @foreach ($group->enabledAgents as $agent)
                                                <x-user-option :user="$agent->user" :selected="(request('agent') == $agent->user->id) || (request('agent') == 'me' && $agent->user->id == user()->id)" />
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
                            <option value="urgent">@lang('modules.tickets.urgent')</option>
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
@include('quality_control::partials.titan-links')

    <!-- CONTENT WRAPPER START -->
    <div class="content-wrapper">
        <div class="row row-cols-lg-5 my-3">

            <div class="col mb-4">
                <a href="javascript:;" data-status="all" class="widget-filter-status">
                    <x-cards.widget :title="__('modules.tickets.totalTickets')" value="0" icon="ticket-alt"
                        widgetId="totalTickets" />
                </a>
            </div>

            <div class="col mb-4">
                <a href="javascript:;" data-status="closed" class="widget-filter-status">
                    <x-cards.widget :title="__('modules.tickets.totalClosedTickets')" value="0" icon="ticket-alt"
                        widgetId="closedTickets" />
                </a>
            </div>

            <div class="col mb-4">
                <a href="javascript:;" data-status="open" class="widget-filter-status">
                    <x-cards.widget :title="__('modules.tickets.totalOpenTickets')" value="0" icon="ticket-alt"
                        widgetId="openTickets" />
                </a>
            </div>

            <div class="col mb-4">
                <a href="javascript:;" data-status="pending" class="widget-filter-status">
                    <x-cards.widget :title="__('modules.tickets.totalPendingTickets')" value="0" icon="ticket-alt"
                        widgetId="pendingTickets" />
                </a>
            </div>

            <div class="col">
                <a href="javascript:;" data-status="resolved" class="widget-filter-status">
                    <x-cards.widget :title="__('modules.tickets.totalResolvedTickets')" value="0" icon="ticket-alt"
                        widgetId="resolvedTickets" />
                </a>
            </div>
        </div>

        <!-- Add Task Export Buttons Start -->
        <div class="d-flex justify-content-between action-bar">
            <div id="table-actions" class="flex-grow-1 align-items-center mt-3">
                 {{-- tombol export --}}
            </div>

            @if (!in_array('client', user_roles()))
                <x-datatable.actions>
                    <div class="select-status mr-3 pl-3">
                        <select name="action_type" class="form-control select-picker" id="quick-action-type" disabled>
                            <option value="">@lang('app.selectAction')</option>
                            <option value="change-status">@lang('modules.tasks.changeStatus')</option>
                            <option value="delete">@lang('app.delete')</option>
                        </select>
                    </div>
                    <div class="select-status mr-3 d-none quick-action-field" id="change-status-action">
                        <select name="status" class="form-control select-picker">
                            <option value="open">@lang('app.open')</option>
                            <option value="pending">@lang('app.pending')</option>
                            <option value="resolved">@lang('app.resolved')</option>
                            <option value="closed">@lang('app.closed')</option>
                        </select>
                    </div>
                </x-datatable.actions>
            @endif
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


            var searchText = $('#search-text-field').val();

            data['startDate'] = startDate;
            data['endDate'] = endDate;
            data['agentId'] = agentId;
            data['priority'] = priority;
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

        $('#agent_id, #ticket-status, #priority')
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


        $('#quick-action-type').change(function() {
            const actionValue = $(this).val();
            if (actionValue != '') {
                $('#quick-action-apply').removeAttr('disabled');

                if (actionValue == 'change-status') {
                    $('.quick-action-field').addClass('d-none');
                    $('#change-status-action').removeClass('d-none');
                } else {
                    $('.quick-action-field').addClass('d-none');
                }
            } else {
                $('#quick-action-apply').attr('disabled', true);
                $('.quick-action-field').addClass('d-none');
            }
        });

        $('#quick-action-apply').click(function() {
            const actionValue = $('#quick-action-type').val();
            if (actionValue == 'delete') {
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
                        applyQuickAction();
                    }
                });

            } else {
                applyQuickAction();
            }
        });

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
                    var url = "{{ route('schedule-inspection.destroy', ':id') }}";
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
            var url = "{{ route('schedule-inspection.change-status') }}";
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
                        scheduleId: id,
                        status: status,
                    },
                    success: function(data) {
                        if(data.status == 'success') {
                            refreshCount();
                        }
                        window.LaravelDataTables["ticket-table"].draw(false);
                    }
                });

            }
        });

        const applyQuickAction = () => {
            var rowdIds = $("#ticket-table input:checkbox:checked").map(function() {
                return $(this).val();
            }).get();

            var url = "{{ route('inspection.apply_quick_action') }}?row_ids=" + rowdIds;

            $.easyAjax({
                url: url,
                container: '#quick-action-form',
                type: "POST",
                disableButton: true,
                buttonSelector: "#quick-action-apply",
                data: $('#quick-action-form').serialize(),
                success: function(response) {
                    if (response.status == 'success') {
                        showTable();
                        resetActionButtons();
                        deSelectAll();
                        $('#quick-action-form').hide();
                    }
                }
            })
        };

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

            // @if (!is_null(request('status')) && !is_null(request('startDate')) && !is_null(request('endDate')))
            //     startDate = '{{ request('startDate') }}';
            //     endDate = '{{ request('endDate') }}';
            // @endif

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



            var url = "{{ route('schedule-inspection.refresh_count') }}";
            $.easyAjax({
                type: 'POST',
                url: url,
                data: {
                    'startDate': startDate,
                    'endDate': endDate,
                    'agentId': agentId,
                    'ticketStatus': status,
                    'priority': priority,
                    '_token': '{{ csrf_token() }}'
                },
                success: function(response) {
                    $('#totalTickets').html(response.totalTickets);
                    $('#closedTickets').html(response.closedTickets);
                    $('#openTickets').html(response.openTickets);
                    $('#pendingTickets').html(response.pendingTickets);
                    $('#resolvedTickets').html(response.resolvedTickets);
                }
            });
        }

        $('body').on('click', '#add-ticket', function() {
            window.location.href = "{{ route('ticket-form.index') }}";
        });

        $( document ).ready(function() {
            @if (!is_null(request('startDate')) && !is_null(request('endDate')))
            $('#datatableRange').val('{{ request('startDate') }}' +
            ' @lang("app.to") ' + '{{ request('endDate') }}');
            $('#datatableRange').data('daterangepicker').setStartDate("{{ request('startDate') }}");
            $('#datatableRange').data('daterangepicker').setEndDate("{{ request('endDate') }}");
                refreshCount();
            @else
            refreshCount();
            @endif
        });
    </script>
@endpush
