@extends('layouts.main')
@section('page-title')
    {{ __('Task') }}
@endsection
@section('page-breadcrumb')
    {{ __('Task') }}
@endsection
@section('page-action')
    <div class="d-flex">
        @permission('task manage')
            <div class="col-sm-auto">
                <a class="btn btn-sm me-2 btn-primary" data-bs-toggle="tooltip" href="{{ route('projects.show', [$id]) }}"
                    data-bs-original-title="{{ __('Project Detail') }}">
                    <i class="ti ti-list"></i>
                </a>
            </div>
        @endpermission
            <div class="col-sm-auto">
                <a class="btn btn-sm me-2 btn-primary" data-bs-toggle="tooltip" href="{{ route('projects.gantt', [$id]) }}"
                    data-bs-original-title="{{ __('Gantt Chart') }}">
                    <i class="ti ti-chart-bar"></i>
                </a>
            </div>
        @permission('task manage')
                <div class="col-sm-auto">
                    <a class="btn btn-sm me-2 btn-primary" data-bs-toggle="tooltip" href="{{ route('projects.task.board', [$id]) }}"
                        data-bs-original-title="{{ __('Task Board') }}">
                        <i class="ti ti-layout-kanban"></i>
                    </a>
                </div>
        @endpermission
        @permission('task create')
        <div class="col-sm-auto">
            <a class="btn btn-sm btn-primary me-2" data-ajax-popup="true" data-size="md"
                data-title="{{ __('Create New Task') }}" data-url="{{ route('tasks.create' ,$id) }}" data-toggle="tooltip"
                title="{{ __('Create') }}">
                <i class="ti ti-plus"></i>
            </a>
        </div>
        @endpermission

        <a href="{{ route('projects.show',$id) }}" class="btn-submit btn btn-sm btn-primary"
            data-toggle="tooltip" title="{{ __('Back') }}">
            <i class=" ti ti-arrow-back-up"></i>
        </a>
    </div>
@endsection
@push('css')
    <link rel="stylesheet" href="{{ asset('packages/workdo/Taskly/src/Resources/assets/css/main.css') }}">
@endpush
@section('content')
    <div class="row">
        <div class="col-sm-12 col-lg-12 col-xl-12 col-md-12">
            <div class="card">
                <div class="card-body">
                    {{ Form::open(['route' => ['projects.calendar' , $id], 'method' => 'get', 'id' => 'task_filter']) }}
                    <div class="d-flex align-items-center justify-content-end">
                        <div class="col-xl-3 col-lg-3 col-md-6 col-sm-12 col-12 mx-2">
                            <div class="btn-box">
                                {{ Form::label('start_date', __('Start Date'), ['class' => 'form-label']) }}
                                {{ Form::date('start_date', isset($_GET['start_date']) ? $_GET['start_date'] : '', ['class' => 'form-control ', 'placeholder' => 'Select Date']) }}
                            </div>
                        </div>
                        <div class="col-xl-3 col-lg-3 col-md-6 col-sm-12 col-12 mx-2">
                            <div class="btn-box">
                                {{ Form::label('due_date', __('Due Date'), ['class' => 'form-label']) }}
                                {{ Form::date('due_date', isset($_GET['due_date']) ? $_GET['due_date'] : '', ['class' => 'form-control ', 'placeholder' => 'Select Date']) }}
                            </div>
                        </div>
                        <div class="col-auto float-end mt-4 d-flex">
                            <a  class="btn btn-sm btn-primary me-2"
                                onclick="document.getElementById('task_filter').submit(); return false;"
                                data-bs-toggle="tooltip" title="" data-bs-original-title="apply">
                                <span class="btn-inner--icon"><i class="ti ti-search"></i></span>
                            </a>
                            <a href="{{ route('projects.calendar' ,$id) }}" class="btn btn-sm btn-danger"
                                data-bs-toggle="tooltip" title="" data-bs-original-title="Reset">
                                <span class="btn-inner--icon"><i class="ti ti-trash-off text-white-off "></i></span>
                            </a>
                        </div>
                    </div>
                    {{ Form::close() }}
                </div>
            </div>
        </div>

        <div class="col-lg-8">
            <div class="card">
                <div class="card-header">
                    <h5>{{ __('Calendar') }}</h5>
                </div>
                <div class="card-body">
                    <div id='calendar' class='calendar'></div>
                </div>
            </div>
        </div>
        <div class="col-lg-4">

            <div class="card">
                <div class="card-body">
                    <h4 class="mb-4">{{ __('Upcoming Tasks') }}</h4>
                    <ul class="event-cards list-group list-group-flush mt-3 w-100">
                        @forelse ($current_month_task as $event)
                            <li class="list-group-item card mb-3">
                                <div class="row align-items-center justify-content-between">
                                    <div class="col-auto mb-3 mb-sm-0">
                                        <div class="d-flex align-items-center">
                                            <div class="theme-avtar badge bg-primary">
                                                <i class="ti ti-calendar-event"></i>
                                            </div>
                                            <div class="ms-3">
                                                <h6 class="card-text small text-primary">{{ $event->title }}</h6>
                                                <div class="card-text small text-dark">{{ __('Start Date :') }}
                                                    {{ company_date_formate($event->start_date) }}
                                                </div>
                                                <div class="card-text small text-dark">{{ __('End Date :') }}
                                                    {{ company_date_formate($event->due_date) }}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </li>
                        @empty
                            <div class="text-center">
                                <h6>{{ __('There is no Task in this month')}}</h6>
                            </div>
                        @endforelse

                    </ul>
                </div>
            </div>
        </div>
    </div>
@endsection
@push('scripts')
    <script src="{{ asset('packages/workdo/Taskly/src/Resources/assets/js/main.min.js') }}"></script>
    <script type="text/javascript">
        (function() {
            var etitle;
            var etype;
            var etypeclass;
            var calendar = new FullCalendar.Calendar(document.getElementById('calendar'), {
                headerToolbar: {
                    left: 'prev,next today',
                    center: 'title',
                    right: 'dayGridMonth,timeGridWeek,timeGridDay'
                },
                buttonText: {
                    today: "{{ __('Today') }}",
                    timeGridDay: "{{ __('Day') }}",
                    timeGridWeek: "{{ __('Week') }}",
                    dayGridMonth: "{{ __('Month') }}"
                },
                themeSystem: 'bootstrap',
                slotDuration: '00:10:00',
                navLinks: true,
                droppable: true,
                selectable: true,
                selectMirror: true,
                editable: true,
                dayMaxEvents: true,
                handleWindowResize: true,
                firstDay: {{ company_setting('calendar_start_day') ?? 0 }},
                events: {!! $arrTask !!},
            });
            calendar.render();
        })();
    </script>
    <script>
        $(document).on('click', '.task-edit', function(e) {
            e.preventDefault();
            var event = $(this);
            var title = $(this).find('.fc-event-title-container .fc-event-title').html();
            var size = 'lg';
            var url = $(this).attr('href');
            $("#commonModal .modal-title").html(title);

            $("#commonModal .modal-dialog").addClass('modal-' + size);
            $.ajax({
                url: url,
                success: function(data) {
                    $('#commonModal .body').html(data);
                    $("#commonModal").modal('show');
                    common_bind();
                    select2();
                },
                error: function(data) {
                    data = data.responseJSON;
                    toastrs('Error', data.error, 'error')
                }
            });
        });
    </script>
@endpush
