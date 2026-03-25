@extends('layouts.admin')

@section('page-title')
    {{ __('Tasks') }}
@endsection
@section('links')
    @if (\Auth::guard('client')->check())
        <li class="breadcrumb-item"><a href="{{ route('client.home') }}">{{ __('Home') }}</a></li>
    @else
        <li class="breadcrumb-item"><a href="{{ route('home') }}">{{ __('Home') }}</a></li>
    @endif
    <li class="breadcrumb-item"> {{ __('Tasks') }}</li>
@endsection
@section('action-button')
    <a href="#" class="btn btn-sm btn-primary filter" data-toggle="tooltip" title="{{ __('Filter') }}">
        <i class="ti ti-filter"></i>
    </a>
@endsection
@push('css-page')
    <style>
        .page-content .select2-container {
            z-index: 0 !important;
        }

        .display-none {
            display: none !important;
        }
    </style>
@endpush

@section('content')
    <div class="row  display-none" id="show_filter">
        <div class=" form-group col-2">
            <select class=" form-select" name="project" id="project">
                <option value="">{{ __('All Projects') }}</option>
                @foreach ($projects as $project)
                    <option value="{{ $project->id }}">{{ $project->name }}</option>
                @endforeach
            </select>
        </div>
        @if ($currentWorkspace->permission == 'Owner')
            <div class="col-2">
                <select class="select2 form-select" name="all_users" id="all_users">
                    <option value="" class="px-4">{{ __('All Users') }}</option>
                    @foreach ($users as $user)
                        <option value="{{ $user->id }}">{{ $user->name }}</option>
                    @endforeach
                </select>
            </div>
        @endif
        <div class="col-auto">
            <select class="select2 form-select" name="status" id="status">
                <option value="" class="px-4">{{ __('All Status') }}</option>
                @foreach ($stages as $stage)
                    <option value="{{ $stage->id }}">{{ __($stage->name) }}</option>
                @endforeach
            </select>
        </div>
        <div class="col-auto">
            <select class="select2 form-select" size="sm" name="priority" id="priority">
                <option value="" class="px-4">{{ __('All Priority') }}</option>
                <option value="Low">{{ __('Low') }}</option>
                <option value="Medium">{{ __('Medium') }}</option>
                <option value="High">{{ __('High') }}</option>
            </select>
        </div>
        <div class="col-auto ">
            <div class='input-group'>
                <input type='text' class=" form-control pc-daterangepicker-2" id="duration1" name="duration"
                    value="{{ __('Select Date Range') }}" placeholder="Select date range" />
                <input type="hidden" name="start_date1" id="start_date1">
                <input type="hidden" name="due_date1" id="end_date1">
                <span class="input-group-text"><i class="feather icon-calendar"></i></span>
            </div>

        </div>
        <div class="col-2">
            <select class="select2 form-select" name="due_date_order" id="due_date_order">
                <option value="due_date,asc " class="px-4">{{ __('Oldest') }}</option>
                <option value="due_date,desc" class="px-4">{{ __('Newest') }}</option>
            </select>
        </div>
        <button class=" btn btn-primary py-2 col-1 btn-filter">{{ __('Apply') }}</button>
    </div>

    <div class="card">

        <div class="card-body mt-3 mx-2">
            <div class="row">
                <div class="col-md-12 mt-2">

                    <div class="table-responsive">
                        <table class="table table-centered table-hover mb-0 animated selection-datatable px-4 mt-2"
                            id="tasks-selection-datatable">
                            <thead>
                                <th>{{ __('Task') }}</th>
                                <th>{{ __('Project') }}</th>
                                <th>{{ __('Milestone') }}</th>
                                <th>{{ __('Due Date') }}</th>
                                @if ($currentWorkspace->permission == 'Owner' || Auth::user()->getGuard() == 'client')
                                    <th>{{ __('Assigned to') }}</th>
                                @endif
                                <th>{{ __('Status') }}</th>
                                <th>{{ __('Priority') }}</th>
                                @if ($currentWorkspace->permission == 'Owner')
                                    <th>{{ __('Action') }}</th>
                                @endif
                            </thead>
                            <tbody>

                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

@endsection

@push('css-page')
@endpush
<link rel="stylesheet" href="{{ asset('custom/css/datatables.min.css') }}">
@push('scripts')
    <script src="{{ asset('custom/js/jquery.dataTables.min.js') }}"></script>
    <script>
        const dataTable = new simpleDatatables.DataTable("#tasks-selection-datatable");
    </script>
    <script type="text/javascript">
        $(".filter").click(function() {
            $("#show_filter").toggleClass('display-none');
        });
    </script>
    <!-- data-picker -->
    <script src="{{ asset('assets/js/plugins/flatpickr.min.js') }}"></script>
    <script>
        document.querySelector(".pc-daterangepicker-2").flatpickr({
            mode: "range"
        });
    </script>
    <script>
        $(function() {
            function cb(start, end) {
                $("#duration1").val(start.format('MMM D, YYYY') + ' - ' + end.format('MMM D, YYYY'));
                $('input[name="start_date1"]').val(start.format('YYYY-MM-DD'));
                $('input[name="due_date1"]').val(end.format('YYYY-MM-DD'));
            }

            $('#duration1').daterangepicker({
                autoApply: true,
                autoclose: true,
                autoUpdateInput: false,
                locale: {
                    format: 'MMM D, YY hh:mm A',
                    applyLabel: "{{ __('Apply') }}",
                    cancelLabel: "{{ __('Cancel') }}",
                    fromLabel: "{{ __('From') }}",
                    toLabel: "{{ __('To') }}",
                    daysOfWeek: [
                        "{{ __('Sun') }}",
                        "{{ __('Mon') }}",
                        "{{ __('Tue') }}",
                        "{{ __('Wed') }}",
                        "{{ __('Thu') }}",
                        "{{ __('Fri') }}",
                        "{{ __('Sat') }}"
                    ],
                    monthNames: [
                        "{{ __('January') }}",
                        "{{ __('February') }}",
                        "{{ __('March') }}",
                        "{{ __('April') }}",
                        "{{ __('May') }}",
                        "{{ __('June') }}",
                        "{{ __('July') }}",
                        "{{ __('August') }}",
                        "{{ __('September') }}",
                        "{{ __('October') }}",
                        "{{ __('November') }}",
                        "{{ __('December') }}"
                    ],
                }
            }, cb);
            // cb(start,end);
        });

        $(document).ready(function() {
            var table = $("#tasks-selection-datatable").DataTable({
                order: [],
                select: {
                    style: "multi"
                },
                "language": e,
                drawCallback: function() {
                    $(".dataTables_paginate > .pagination").addClass("pagination-rounded")
                }
            });

            $(document).on("click", ".btn-filter", function() {
                getData();
            });

            function getData() {
                table.clear().draw();
                $("#tasks-selection-datatable tbody tr").html(
                    '<td colspan="11" class="text-center"> {{ __('Loading ...') }}</td>');

                var data = {
                    project: $("#project").val(),
                    assign_to: $("#all_users").val(),
                    priority: $("#priority").val(),
                    due_date_order: $("#due_date_order").val(),
                    status: $("#status").val(),
                    start_date: $("#start_date1").val(),
                    end_date: $("#end_date1").val(),
                };

                $.ajax({
                    url: '{{ route('tasks.ajax', [$currentWorkspace->slug]) }}',
                    type: 'POST',
                    data: data,
                    success: function(data) {

                        table.rows.add(data.data).draw();
                        loadConfirm();
                    },
                    error: function(data) {
                        toastrs('Info', data.error, 'info')
                    }
                })
            }

            getData();

        });
    </script>
@endpush
