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

        @if (!in_array('client', user_roles()))
            <!-- CLIENT START -->
            <div class="select-box d-flex py-2 px-lg-2 px-md-2 px-0 border-right-grey border-right-grey-sm-0">
                <p class="mb-0 pr-2 f-14 text-dark-grey d-flex align-items-center">@lang('app.client')</p>
                <div class="select-status">
                    <select class="form-control select-picker" id="clientID" data-live-search="true" data-size="8">
                        @if (!in_array('client', user_roles()))
                            <option value="all">@lang('app.all')</option>
                        @endif
                        @foreach ($clients as $client)
                                <x-user-option :user="$client" />
                            </option>
                        @endforeach
                    </select>
                </div>
            </div>
            <!-- CLIENT END -->
        @endif

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
            <div class="more-filter-items">
                <label class="f-14 text-dark-grey mb-12 " for="usr">@lang('app.project')</label>
                <div class="select-filter mb-4">
                    <div class="select-others">
                        <select class="form-control select-picker" name="project_id" id="filter_project_id"
                            data-container="body" data-live-search="true" data-size="8">
                            <option value="all">@lang('app.all')</option>
                            @foreach ($projects as $project)
                                <option value="{{ $project->id }}">{{ $project->project_name }}</option>
                            @endforeach
                        </select>
                    </div>
                </div>
            </div>

            <div class="more-filter-items">
                <label class="f-14 text-dark-grey mb-12 " for="usr">@lang('app.status')</label>
                <div class="select-filter mb-4">
                    <div class="select-others">
                        <select class="form-control select-picker" name="status" id="status" data-live-search="true"
                            data-container="body" data-size="8">
                            <option value="all">@lang('app.all')</option>
                            <option {{ request('status') == 'pending' ? 'selected' : '' }} value="pending">
                                @lang('app.pending')</option>
                            <option {{ request('status') == 'unpaid' ? 'selected' : '' }} value="unpaid">
                                @lang('app.unpaid')</option>
                            <option {{ request('status') == 'paid' ? 'selected' : '' }} value="paid">@lang('app.paid')
                            </option>
                            <option {{ request('status') == 'partial' ? 'selected' : '' }} value="partial">
                                @lang('app.partial')</option>
                            <option {{ request('status') == 'canceled' ? 'selected' : '' }} value="canceled">
                                @lang('app.canceled')</option>
                        </select>
                    </div>
                </div>
            </div>


        </x-filters.more-filter-box>
        <!-- MORE FILTERS END -->

    </x-filters.filter-box>

@endsection

@php
$addInvoicesPermission = user()->permission('add_invoices');
$manageRecurringInvoicesPermission = user()->permission('manage_recurring_invoice');
@endphp

@section('content')
    <!-- CONTENT WRAPPER START -->
    <div class="content-wrapper">
        @if(user()->permission('manage_finance_setting') == 'all' && (in_array('invoices', user_modules())))
            <x-alert type="primary">
                <span class="mb-12"><strong>Note:</strong></span>
                <span>@lang('einvoice::app.settingsNote') <a href="javascript:;" class="einvoice-setting">@lang('app.settings')</a></span>
            </x-alert>
        @endif
        <!-- Add Task Export Buttons Start -->
        <div class="d-block d-lg-flex d-md-flex justify-content-between">
            <div id="table-actions" class="flex-grow-1 align-items-center mb-2 mb-lg-0 mb-md-0">


            </div>
            @if(user()->permission('manage_finance_setting') == 'all' && (in_array('invoices', user_modules())))
                <div class="btn-group mt-3 mt-lg-0 pb-1 mt-md-0 ml-lg-3 d-none d-lg-block" role="group">
                    <button class="btn btn-secondary" id="einvoice-setting" type="button"><span><i class="fa fa-cog"></i> @lang('app.settings')</span></button>
                </div>
            @endif
        </div>

        <!-- Add Task Export Buttons End -->
        <!-- Task Box Start -->
        <div class="d-flex flex-column w-tables rounded mt-3 bg-white w-100 table-responsive">

            {!! $dataTable->table(['class' => 'table table-hover border-0 w-100']) !!}

        </div>
        <!-- Task Box End -->
    </div>
    <!-- CONTENT WRAPPER END -->

@endsection

@push('scripts')
    @include('sections.datatable_js')
    <script>
        $(function() {

            $('#invoices-table').on('preXhr.dt', function(e, settings, data) {

                var dateRangePicker = $('#datatableRange').data('daterangepicker');
                var startDate = $('#datatableRange').val();

                if (startDate == '') {
                    startDate = null;
                    endDate = null;
                } else {
                    startDate = dateRangePicker.startDate.format('{{ company()->moment_date_format }}');
                    endDate = dateRangePicker.endDate.format('{{ company()->moment_date_format }}');
                }

                var projectID = $('#filter_project_id').val();
                if (!projectID) {
                    projectID = 0;
                }
                var clientID = $('#clientID').val();
                var status = $('#status').val();

                var searchText = $('#search-text-field').val();

                data['clientID'] = clientID;
                data['projectID'] = projectID;
                data['status'] = status;
                data['startDate'] = startDate;
                data['endDate'] = endDate;
                data['searchText'] = searchText;
            });

            const showTable = () => {
                window.LaravelDataTables["invoices-table"].draw(true);
            }

            $('#clientID, #filter_project_id, #status')
                .on('change keyup',
                    function() {
                        if ($('#filter_project_id').val() != "all") {
                            $('#reset-filters').removeClass('d-none');
                            showTable();
                        } else if ($('#status').val() != "all") {
                            $('#reset-filters').removeClass('d-none');
                            showTable();
                        } else if ($('#clientID').val() != "all") {
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

            $('#reset-filters,#reset-filters-2').click(function () {
                $('#filter-form')[0].reset();

                $('.filter-box .select-picker').selectpicker("refresh");
                $('#reset-filters').addClass('d-none');
                showTable();
            });

            $('body').on('click', '#einvoice-setting, .einvoice-setting', function() {
                $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
                $.ajaxModal(MODAL_LG, "{{ route('einvoice.settings_modal') }}");
            });

            $('body').on('click', '.downloadInvoice', function() {
                if ($(this).hasClass('disabled')) {
                   return false;
                }

                var id = $(this).data('id');
                var url = "{{ route('einvoice.exportXml', ':id') }}";
                url = url.replace(':id', id);
                window.location.href = url;
            });

            $('body').on('click', '.updateClient', function() {
                var id = $(this).data('id');
                var url = "{{ route('einvoice.client_modal', ':id') }}";
                url = url.replace(':id', id);
                $(MODAL_LG + ' ' + MODAL_HEADING).html('...');
                $.ajaxModal(MODAL_LG, url);
            });

            @if (request('start') && request('end'))
                $('#datatableRange').val('{{ request('start') }}' +
                ' @lang("app.to") ' + '{{ request('end') }}');
                $('#datatableRange').data('daterangepicker').setStartDate("{{ request('start') }}");
                $('#datatableRange').data('daterangepicker').setEndDate("{{ request('end') }}");
                showTable();
            @endif

            @if (session('message'))
                Swal.fire({
                    icon: 'error',
                    text: '{{ session('message') }}',
                    showConfirmButton: true,
                    customClass: {
                        confirmButton: 'btn btn-primary',
                    },
                    showClass: {
                        popup: 'swal2-noanimation',
                        backdrop: 'swal2-noanimation'
                    },
                })
            @endif
    });
    </script>
@endpush
