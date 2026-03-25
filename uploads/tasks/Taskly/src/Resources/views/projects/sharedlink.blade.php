@php
    $setting = [];
    if ($project->copylinksetting) {
        $setting = json_decode($project->copylinksetting);
    }

    $title =
        module_is_active('ProjectTemplate') && $project->type == 'template' ? __('Project Template') : __('Project');
@endphp
@extends('layouts.invoicepayheader')
@section('page-title')
    {{ $title . ' ' . $project->name }}
@endsection
@section('language-bar')
@endsection
@push('css')
    <link rel="stylesheet" href="{{ asset('packages/workdo/Taskly/src/Resources/assets/css/custom.css') }}" type="text/css" />
    <link rel="stylesheet" href="{{ asset('assets/css/plugins/dragula.min.css') }}">
@endpush
@section('action-btn')
    <a class="dash-head-link dropdown-toggle btn btn-primary text-white" data-bs-toggle="dropdown" href="#">
        <span class="drp-text hide-mob text-white">{{ Str::upper($lang) }}</span>
    </a>
    <div class="dropdown-menu dash-h-dropdown dropdown-menu-end">
        @foreach (languages() as $key => $language)
            <a href="{{ route('project.shared.link', [$id, $key]) }}"
                class="dropdown-item @if ($lang == $key) text-primary @endif">
                <span>{{ Str::ucfirst($language) }}</span>
            </a>
        @endforeach
    </div>
@endsection
@section('content')
    <div class="row">
        <div class="col-xl-3">
            <div class="card sticky-top" style="top:30px">
                <div class="list-group list-group-flush" id="useradd-sidenav">
                    @if (isset($setting->basic_details) && $setting->basic_details == 'on')
                        <a href="#tabs-1" class="list-group-item list-group-item-action border-0">{{ __('Basic details') }}
                            <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                        </a>
                    @endif
                    @if (isset($setting->progress) && $setting->progress == 'on')
                        <a href="#tabs-11" class="list-group-item list-group-item-action border-0">{{ __('Progress') }}
                            <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                        </a>
                    @endif
                    @if (isset($setting->member) && $setting->member == 'on')
                        <a href="#tabs-2" class="list-group-item list-group-item-action border-0 ">{{ __('Members') }} <div
                                class="float-end"><i class="ti ti-chevron-right"></i></div></a>
                    @endif
                    @if (isset($setting->client) && $setting->client == 'on')
                        <a href="#tabs-3" class="list-group-item list-group-item-action border-0">{{ __('Clients') }} <div
                                class="float-end"><i class="ti ti-chevron-right"></i></div></a>
                    @endif
                    @if (module_is_active('Account'))
                        @if (isset($setting->vendor) && $setting->vendor == 'on')
                            <a href="#tabs-31" class="list-group-item list-group-item-action border-0">{{ __('Vendors') }}
                                <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                            </a>
                        @endif
                    @endif
                    @if (isset($setting->milestone) && $setting->milestone == 'on')
                        <a href="#tabs-4" class="list-group-item list-group-item-action border-0">{{ __('Milestones') }}
                            <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                        </a>
                    @endif
                    @if (isset($setting->attachment) && $setting->attachment == 'on')
                        <a href="#tabs-5" class="list-group-item list-group-item-action border-0">{{ __('Files') }} <div
                                class="float-end"><i class="ti ti-chevron-right"></i></div></a>
                    @endif
                    @if (isset($setting->task) && $setting->task == 'on')
                        <a href="#tabs-6" class="list-group-item list-group-item-action border-0">{{ __('Task') }} <div
                                class="float-end"><i class="ti ti-chevron-right"></i></div></a>
                    @endif
                    @if (isset($setting->bug_report) && $setting->bug_report == 'on')
                        <a href="#tabs-7" class="list-group-item list-group-item-action border-0">{{ __('Bug Report') }}
                            <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                        </a>
                    @endif
                    @if (isset($setting->invoice) && $setting->invoice == 'on')
                        <a href="#tabs-7-1" class="list-group-item list-group-item-action border-0">{{ __('Invoice') }}
                            <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                        </a>
                    @endif
                    @if (module_is_active('Account'))
                        @if (isset($setting->bill) && $setting->bill == 'on')
                            <a href="#tabs-7-1-2"
                                class="list-group-item list-group-item-action border-0">{{ __('Bill') }}
                                <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                            </a>
                        @endif
                    @endif
                    @if (module_is_active('Timesheet', $project->created_by))
                        @if (isset($setting->timesheet) && $setting->timesheet == 'on')
                            <a href="#tabs-8" class="list-group-item list-group-item-action border-0">{{ __('Timesheet') }}
                                <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                            </a>
                        @endif
                    @endif
                    @if (isset($setting->activity) && $setting->activity == 'on')
                        <a href="#tabs-9" class="list-group-item list-group-item-action border-0">{{ __('Activity Log') }}
                            <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                        </a>
                    @endif
                    @if (module_is_active('Documents', $setting->documents ?? 'on'))
                        @if (isset($setting->documents) && $setting->documents == 'on')
                            <a href="#tabs-10"
                                class="list-group-item list-group-item-action border-0">{{ __('Documents') }}
                                <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                            </a>
                        @endif
                    @endif
                    @if (module_is_active('Retainer'))
                        @if (isset($setting->retainer) && $setting->retainer == 'on')
                            <a href="#tabs-12" class="list-group-item list-group-item-action border-0">{{ __('Retainer') }}
                                <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                            </a>
                        @endif
                    @endif
                    @if (isset($setting->proposal) && $setting->proposal == 'on')
                        <a href="#tabs-13" class="list-group-item list-group-item-action border-0">{{ __('Proposal') }}
                            <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                        </a>
                    @endif
                    @if (module_is_active('Procurement'))
                        @if (isset($setting->procurement) && $setting->procurement == 'on')
                            <a href="#tabs-14"
                                class="list-group-item list-group-item-action border-0">{{ __('Procurement') }}
                                <div class="float-end"><i class="ti ti-chevron-right"></i></div>
                            </a>
                        @endif
                    @endif
                </div>
            </div>
        </div>
        <div class="col-xl-9">

            @if (isset($setting->basic_details) && $setting->basic_details == 'on')
                <div id="tabs-1" class="">
                    <div class="card bg-primary">
                        <div class="card-body">
                            <div class="d-block d-sm-flex align-items-center justify-content-between">
                                <h4 class="text-white"> {{ $project->name }}</h4>
                                <div class="d-flex  align-items-center row1">
                                    @if ($project->type == 'project')
                                        <div class="px-3">
                                            <span class="text-white text-sm">{{ __('Start Date') }}:</span>
                                            <h5 class="text-white text-nowrap">
                                                {{ company_date_formate($project->start_date, $project->created_by, $project->workspace) }}
                                            </h5>
                                        </div>
                                        <div class="px-3">
                                            <span class="text-white text-sm">{{ __('Due Date') }}:</span>
                                            <h5 class="text-white">
                                                {{ company_date_formate($project->end_date, $project->created_by, $project->workspace) }}
                                            </h5>
                                        </div>
                                        <div class="px-3">
                                            <span class="text-white text-sm">{{ __('Total Members') }}:</span>
                                            <h5 class="text-white text-nowrap">
                                                {{ (int) $project->users->count() + (int) $project->clients->count() }}
                                            </h5>
                                        </div>
                                    @endif
                                    <div class="px-3 py-3">

                                        @if ($project->status == 'Finished')
                                            <div class="badge  bg-success p-2 px-3"> {{ __('Finished') }}
                                            </div>
                                        @elseif($project->status == 'Ongoing')
                                            <div class="badge  bg-secondary p-2 px-3">{{ __('Ongoing') }}</div>
                                        @else
                                            <div class="badge bg-warning p-2 px-3">{{ __('OnHold') }}</div>
                                        @endif

                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        @if ($project->type == 'project')
                            <div class="col-lg-4 col-sm-6">
                                <div class="card">
                                    <div class="card-body">
                                        <div class="row">
                                            <div class="theme-avtar badge bg-primary">
                                                <i class="fas fas fa-calendar-day"></i>
                                            </div>
                                            <div class="col text-end">
                                                <h6 class="text-muted mb-1">{{ __('Days left') }}</h6>
                                                <span class="h6 font-weight-bold mb-0 ">{{ $daysleft }}</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        @endif
                        @php
                            $class = $project->type == 'template' ? 'col-lg-6 col-sm-6' : 'col-lg-4 col-sm-6';
                        @endphp
                        <div class="{{ $class }}">
                            <div class="card">
                                <div class="card-body">
                                    <div class="row">
                                        <div class="theme-avtar badge bg-danger">
                                            <i class="fas fa-check-double"></i>
                                        </div>
                                        <div class="col text-end">
                                            <h6 class="text-muted mb-1">{{ __('Total Task') }}</h6>
                                            <span class="h6 font-weight-bold mb-0 ">{{ $project->countTask() }}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="{{ $class }}">
                            <div class="card">
                                <div class="card-body">
                                    <div class="row">
                                        <div class="theme-avtar badge bg-success">
                                            <i class="fas fa-comments"></i>
                                        </div>
                                        <div class="col text-end">
                                            <h6 class="text-muted mb-1">{{ __('Comment') }}</h6>
                                            <span
                                                class="h6 font-weight-bold mb-0 ">{{ $project->countTaskComments() }}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            @endif
            @if (isset($setting->progress) && $setting->progress == 'on')
                <div id="tabs-11" class="">
                    <div class="card">
                        <div class="card-header" style="padding: 25px 35px !important;">
                            <div class="d-flex justify-content-between align-items-center">
                                <div class="row">
                                    <h5 class="mb-0">{{ __('Progress') }}<span class="text-end">
                                            {{ __('(Last Week Tasks)') }} </span></h5>
                                </div>
                            </div>
                        </div>
                        <div class="card-body">
                            <div id="task-chart"></div>
                        </div>
                    </div>
                </div>
            @endif
            <div class="row">
                @if (isset($setting->member) && $setting->member == 'on')
                    <div id="tabs-2"
                        class="{{ module_is_active('Account', $project->created_by) && (isset($setting->vendor) && $setting->vendor == 'on') ? 'col-md-4' : 'col-md-6' }}">
                        <div class="card deta-card">
                            <div class="card-header">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <h5 class="mb-0">{{ __('Team Members') }} ({{ count($project->users) }})
                                        </h5>
                                    </div>
                                </div>
                            </div>
                            <div class="card-body">
                                @foreach ($project->users as $user)
                                    <ul class="list-group list-group-flush">
                                        <li class="list-group-item px-0">
                                            <div class="row align-items-center justify-content-between">
                                                <div class="col-sm-auto mb-3 mb-sm-0">
                                                    <div class="d-flex align-items-center px-2">
                                                        <a href="#" class=" text-start">
                                                            <img alt="image" data-bs-toggle="tooltip"
                                                                data-bs-placement="top" title="{{ $user->name }}"
                                                                @if ($user->avatar) src="{{ get_file($user->avatar) }}" @else src="{{ get_file('avatar.png') }}" @endif
                                                                class="rounded-circle " width="40px" height="40px">
                                                        </a>
                                                        <div class="px-2">
                                                            <h5 class="m-0">{{ $user->name }}</h5>
                                                            <small class="text-muted">{{ $user->email }}<span
                                                                    class="text-primary "> -
                                                                    {{ (int) count($project->user_done_tasks($user->id)) }}/{{ (int) count($project->user_tasks($user->id)) }}</span></small>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </li>
                                    </ul>
                                @endforeach
                            </div>
                        </div>
                    </div>
                @endif
                @if (isset($setting->client) && $setting->client == 'on')
                    <div id="tabs-3"
                        class="{{ module_is_active('Account', $project->created_by) && (isset($setting->vendor) && $setting->vendor == 'on') ? 'col-md-4' : 'col-md-6' }}">
                        <div class="card deta-card">

                            <div class="card-header">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <h5 class="mb-0">{{ __('Clients') }} ({{ count($project->clients) }})</h5>
                                    </div>
                                </div>
                            </div>
                            <div class="card-body">
                                @foreach ($project->clients as $client)
                                    <ul class="list-group list-group-flush">
                                        <li class="list-group-item px-0">
                                            <div class="row align-items-center justify-content-between">
                                                <div class="col-sm-auto mb-3 mb-sm-0">
                                                    <div class="d-flex align-items-center px-2">
                                                        <a href="#" class=" text-start">
                                                            <img alt="image" data-bs-toggle="tooltip"
                                                                data-bs-placement="top" title="{{ $client->name }}"
                                                                @if ($client->avatar) src="{{ get_file($client->avatar) }}" @else src="{{ get_file('avatar.png') }}" @endif
                                                                class="rounded-circle " width="40px" height="40px">
                                                        </a>
                                                        <div class="px-2">
                                                            <h5 class="m-0">{{ $client->name }}</h5>
                                                            <small class="text-muted">{{ $client->email }}</small>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </li>
                                    </ul>
                                @endforeach
                            </div>
                        </div>
                    </div>
                @endif
                @if (module_is_active('Account'))
                    @if (isset($setting->vendor) && $setting->vendor == 'on')
                        <div id="tabs-31" class="col-md-4">
                            <div class="card deta-card">
                                <div class="card-header ">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <div>
                                            <h5 class="mb-0">{{ __('Vendors') }} ({{ count($project->venders) }})</h5>
                                        </div>
                                    </div>
                                </div>
                                <div class="card-body top-10-scroll">
                                    @foreach ($project->venders as $client)
                                        <ul class="list-group list-group-flush" style="width: 100%;">
                                            <li class="list-group-item px-0">
                                                <div class="row align-items-center justify-content-between">
                                                    <div class="col-sm-auto mb-3 mb-sm-0">
                                                        <div class="d-flex align-items-center px-2">
                                                            <a href="#" class=" text-start">
                                                                <img alt="image" data-bs-toggle="tooltip"
                                                                    data-bs-placement="top" title="{{ $client->name }}"
                                                                    @if ($client->avatar) src="{{ get_file($client->avatar) }}" @else src="{{ get_file('avatar.png') }}" @endif
                                                                    class="rounded-circle " width="40px"
                                                                    height="40px">
                                                            </a>
                                                            <div class="px-2">
                                                                <h5 class="m-0">{{ $client->name }}</h5>
                                                                <small class="text-muted">{{ $client->email }}</small>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </li>
                                        </ul>
                                    @endforeach
                                </div>
                            </div>
                        </div>
                    @endif
                @endif
            </div>
            @if (isset($setting->milestone) && $setting->milestone == 'on')
                <div id="tabs-4" class="">

                    <div class="card milestone-card">
                        <div class="card-header">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h5 class="mb-0">{{ __('Milestones') }} ({{ count($project->milestones) }})</h5>
                                </div>
                                <div class="float-end">
                                    @permission('milestone create')
                                        <p class="text-muted d-sm-flex align-items-center mb-0">
                                            <a class="btn btn-sm btn-primary" data-size="lg" data-ajax-popup="true"
                                                data-title="{{ __('Create Milestone') }}"
                                                data-url="{{ route('projects.milestone', [$project->id]) }}"
                                                data-toggle="tooltip" title="{{ __('Create') }}"><i
                                                    class="ti ti-plus"></i></a>
                                        </p>
                                    @endpermission
                                </div>
                            </div>
                        </div>
                        <div class="card-body top-10-scroll">

                            <div class="table-responsive">
                                <table id="" class="table table-bordered px-2">
                                    <thead>
                                        <tr>
                                            <th>{{ __('Name') }}</th>
                                            <th>{{ __('Status') }}</th>
                                            <th>{{ __('Start Date') }}</th>
                                            <th>{{ __('End Date') }}</th>
                                            <th>{{ __('Cost') }}</th>
                                            <th>{{ __('Progress') }}</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        @foreach ($project->milestones as $key => $milestone)
                                            <tr>
                                                <td>
                                                    <a href="#" class="d-block font-weight-500 mb-0"
                                                        @permission('milestone delete') data-ajax-popup="true" data-title="{{ __('Milestone Details') }}"  data-url="{{ route('projects.milestone.show', [$milestone->id]) }}" @endpermission>
                                                        <h5 class="m-0"> {{ $milestone->title }} </h5>
                                                    </a>
                                                </td>
                                                <td>

                                                    @if ($milestone->status == 'complete')
                                                        <label
                                                            class="badge bg-success p-2 px-3">{{ __('Complete') }}</label>
                                                    @else
                                                        <label
                                                            class="badge bg-warning p-2 px-3">{{ __('Incomplete') }}</label>
                                                    @endif
                                                </td>
                                                <td>{{ $milestone->start_date }}</td>
                                                <td>{{ $milestone->end_date }}</td>
                                                <td>{{ company_setting('defult_currancy', $project->created_by, $project->workspace) }}{{ $milestone->cost }}
                                                </td>
                                                <td>
                                                    <div class="progress_wrapper">
                                                        <div class="progress">
                                                            <div class="progress-bar" role="progressbar"
                                                                style="width: {{ $milestone->progress }}px;"
                                                                aria-valuenow="55" aria-valuemin="0" aria-valuemax="100">
                                                            </div>
                                                        </div>
                                                        <div class="progress_labels">
                                                            <div class="total_progress">

                                                                <strong> {{ $milestone->progress }}%</strong>
                                                            </div>

                                                        </div>
                                                    </div>
                                                </td>
                                            </tr>
                                        @endforeach
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
            @endif
            @if (isset($setting->attachment) && $setting->attachment == 'on')
                <div id="tabs-5" class="">
                    <div class="card">
                        <div class="card-header">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h5 class="mb-0">{{ __('Files') }}</h5>
                                </div>
                            </div>
                        </div>
                        <div class="card-body p-4">

                            <div class="author-box-name form-control-label mb-4">

                            </div>
                            <div class="col-md-12 dropzone browse-file" id="dropzonewidget11">
                                @php
                                    $files = $project->files;
                                @endphp
                                @foreach ($files as $file)
                                    <img src="{{ get_file($file->file_path) }}" alt=""
                                        class="dropzone_uploaded">
                                    <a href="{{ get_file($file->file_path) }}" download="download"
                                        class="action-btn dropzone_btn btn-primary mx-1  btn btn-sm d-inline-flex align-items-center"><i
                                            class="ti ti-download"> </i></a>
                                @endforeach
                            </div>
                        </div>
                    </div>
                </div>
            @endif
            @if (isset($setting->task) && $setting->task == 'on')
                <div id="tabs-6" class="">
                    <div class="card" style="background-color:transparent !important">
                        <div class="card-header"
                            style="padding: 25px 35px !important; background-color:#ffffff !important">
                            <div class="d-flex justify-content-between align-items-center">
                                <div class="row">
                                    <h5 class="mb-0">{{ __('Task') }}</h5>
                                </div>
                            </div>
                        </div>
                        <div class="card-body">
                            <section class="section">
                                @if ($project)
                                    <div class="row">
                                        <div class="col-sm-12">
                                            <div class="row kanban-wrapper horizontal-scroll-cards pt-3" data-toggle="dragula">
                                                @foreach ($stages as $stage)
                                                    <div class="col" id="backlog">
                                                        <div class="card card-list">
                                                            <div class="card-header">
                                                                <div class="float-end">
                                                                    <button
                                                                        class="btn-submit btn btn-md btn-primary btn-icon px-1  py-0">
                                                                        <span
                                                                            class="badge badge-secondary rounded-pill count">{{ $stage->tasks->count() }}</span>
                                                                    </button>
                                                                </div>
                                                                <h4 class="mb-0">{{ $stage->name }}</h4>
                                                            </div>
                                                            <div id="{{ 'task-list-' . str_replace(' ', '_', $stage->id) }}"
                                                                data-status="{{ $stage->id }}"
                                                                class="card-body kanban-box">
                                                                @foreach ($stage->tasks as $task)
                                                                    <div class="card" id="{{ $task->id }}">
                                                                        <div
                                                                            class="position-absolute top-0 start-0 pt-3 ps-3">
                                                                            @if ($task->priority == 'Low')
                                                                                <div
                                                                                    class="badge bg-success p-2 px-3">
                                                                                    {{ $task->priority }}
                                                                                </div>
                                                                            @elseif($task->priority == 'Medium')
                                                                                <div
                                                                                    class="badge bg-warning p-2 px-3">
                                                                                    {{ $task->priority }}
                                                                                </div>
                                                                            @elseif($task->priority == 'High')
                                                                                <div
                                                                                    class="badge bg-danger p-2 px-3">
                                                                                    {{ $task->priority }}
                                                                                </div>
                                                                            @endif
                                                                        </div>
                                                                        <div
                                                                            class="card-header border-0 pb-0 position-relative">
                                                                            <div style="padding: 30px 2px;">
                                                                                <a href="#"
                                                                                    data-url="{{ route('Project.link.task.show', [$task->project_id, $task->id]) }}"
                                                                                    data-size="lg" data-ajax-popup="true"
                                                                                    data-title="{{ __('Task Detail') }}"
                                                                                    class="h6 task-title">
                                                                                    <h5>{{ $task->title }}</h5>
                                                                                </a>
                                                                            </div>
                                                                        </div>
                                                                        <div class="card-body pt-0">
                                                                            <div class="row">
                                                                                <div class="col">
                                                                                    <div class="action-item">
                                                                                        {{ company_date_formate($task->start_date, $project->created_by, $project->workspace) }}
                                                                                    </div>
                                                                                </div>
                                                                                <div class="col text-end">
                                                                                    <div class="action-item">
                                                                                        {{ company_date_formate($task->due_date, $project->created_by, $project->workspace) }}
                                                                                    </div>
                                                                                </div>
                                                                            </div>
                                                                            <div
                                                                                class="d-flex align-items-center justify-content-between">
                                                                                <ul class="list-inline mb-0">
                                                                                    <li
                                                                                        class="list-inline-item d-inline-flex align-items-center">
                                                                                        <i
                                                                                            class="f-16 text-primary ti ti-brand-telegram"></i>
                                                                                        {{ $task->taskCompleteSubTaskCount() }}/{{ $task->taskTotalSubTaskCount() }}
                                                                                    </li>
                                                                                </ul>
                                                                                <div class="user-group">
                                                                                    @if ($users = $task->users())
                                                                                        @foreach ($users as $key => $user)
                                                                                            @if ($key < 3)
                                                                                                <img alt="image"
                                                                                                    data-bs-toggle="tooltip"
                                                                                                    data-bs-placement="top"
                                                                                                    title="{{ $user->name }}"
                                                                                                    @if ($user->avatar) src="{{ get_file($user->avatar) }}" @else src="{{ get_file('avatar.png') }}" @endif
                                                                                                    class="rounded-circle "
                                                                                                    width="40px"
                                                                                                    height="40px">
                                                                                            @endif
                                                                                        @endforeach
                                                                                        @if (count($users) > 3)
                                                                                            <img alt="image"
                                                                                                data-toggle="tooltip"
                                                                                                data-original-title="{{ count($users) - 3 }} {{ __('more') }}"
                                                                                                avatar="+ {{ count($users) - 3 }}">
                                                                                        @endif
                                                                                    @endif
                                                                                </div>
                                                                            </div>
                                                                        </div>
                                                                    </div>
                                                                @endforeach
                                                                <span class="empty-container"
                                                                    data-placeholder="Empty"></span>
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
                                                            <p class="text-muted mt-3">
                                                                {{ __("It's looking like you may have taken a wrong turn. Don't worry... it happens to the best of us. Here's a little tip that might help you get back on track.") }}
                                                            </p>
                                                            <div class="mt-3">
                                                                <a class="btn-return-home badge-blue"
                                                                    href="{{ route('home') }}"><i
                                                                        class="fas fa-reply"></i>
                                                                    {{ __('Return Home') }}</a>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                @endif
                            </section>
                        </div>
                    </div>
                </div>
            @endif
            @if (isset($setting->bug_report) && $setting->bug_report == 'on')
                <div id="tabs-7" class="">
                    <div class="card" style="background-color:transparent !important">
                        <div class="card-header"
                            style="padding: 25px 35px !important; background-color:#ffffff !important">
                            <div class="d-flex justify-content-between align-items-center">
                                <div class="row">
                                    <h5 class="mb-0">{{ __('Bug Report') }}</h5>

                                </div>
                            </div>
                        </div>
                        <div class="card-body">
                            <section class="section">
                                @if ($project)
                                    <div class="row">
                                        <div class="col-sm-12">
                                            <div class="row kanban-wrapper horizontal-scroll-cards pt-3" data-toggle="dragula">
                                                @foreach ($bug_stages as $stage)
                                                    <div class="col" id="backlog">
                                                        <div class="card card-list">
                                                            <div class="card-header">
                                                                <div class="float-end">
                                                                    <button
                                                                        class="btn-submit btn btn-md btn-primary btn-icon px-1  py-0">
                                                                        <span
                                                                            class="badge badge-secondary rounded-pill count">{{ $stage->bugs->count() }}</span>
                                                                    </button>
                                                                </div>
                                                                <h4 class="mb-0">{{ $stage->name }}</h4>
                                                            </div>
                                                            <div id="{{ 'task-list-' . str_replace(' ', '_', $stage->id) }}"
                                                                data-status="{{ $stage->id }}"
                                                                class="card-body kanban-box">
                                                                @foreach ($stage->bugs as $bug)
                                                                    <div class="card" id="{{ $bug->id }}">
                                                                        <div
                                                                            class="position-absolute top-0 start-0 pt-3 ps-3">
                                                                            @if ($bug->priority == 'Low')
                                                                                <div
                                                                                    class="badge bg-success p-2 px-3">
                                                                                    {{ $bug->priority }}</div>
                                                                            @elseif($bug->priority == 'Medium')
                                                                                <div
                                                                                    class="badge bg-warning p-2 px-3">
                                                                                    {{ $bug->priority }}</div>
                                                                            @elseif($bug->priority == 'High')
                                                                                <div
                                                                                    class="badge bg-danger p-2 px-3">
                                                                                    {{ $bug->priority }}</div>
                                                                            @endif
                                                                        </div>
                                                                        <div
                                                                            class="card-header border-0 pb-0 position-relative">
                                                                            <div style="padding: 30px 2px;">
                                                                                <a href="#" data-size="lg"
                                                                                    data-url="{{ route('projects.link.bug.report.show', [$bug->project_id, $bug->id]) }}"
                                                                                    data-ajax-popup="true"
                                                                                    data-title="{{ __('Bug Detail') }}"
                                                                                    class="h6 task-title">
                                                                                    <h5>{{ $bug->title }}</h5>
                                                                                </a>
                                                                            </div>
                                                                        </div>
                                                                        <div class="card-body pt-0">
                                                                            <div
                                                                                class="d-flex align-items-center justify-content-between">
                                                                                <ul class="list-inline mb-0">
                                                                                    <li
                                                                                        class="list-inline-item d-inline-flex align-items-center">
                                                                                        <i
                                                                                            class="f-16 text-primary ti ti-message-2"></i>
                                                                                        {{ $bug->comments->count() }}
                                                                                        {{ __('Comments') }}
                                                                                    </li>
                                                                                </ul>

                                                                                <div class="user-group">
                                                                                    <a href="#" class="img_group">
                                                                                        <img alt="image"
                                                                                            data-bs-toggle="tooltip"
                                                                                            data-bs-placement="top"
                                                                                            title="{{ !empty($bug->user->name) ? $bug->user->name : '' }}"
                                                                                            @if ($bug->user) src="{{ get_file($bug->user->avatar) }}" @else src="{{ get_file('uploads/users-avatar/avatar.png') }}" @endif
                                                                                            class="rounded-circle "
                                                                                            width="40px" height="40px">
                                                                                    </a>
                                                                                </div>
                                                                            </div>
                                                                        </div>
                                                                    </div>
                                                                @endforeach
                                                                <span class="empty-container"
                                                                    data-placeholder="Empty"></span>
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
                                                            <p class="text-muted mt-3">
                                                                {{ __("It's looking like you may have taken a wrong turn. Don't worry... it happens to the best of us. Here's a little tip that might help you get back on track.") }}
                                                            </p>
                                                            <div class="mt-3">
                                                                <a class="btn-return-home badge-blue"
                                                                    href="{{ route('home') }}"><i
                                                                        class="fas fa-reply"></i>
                                                                    {{ __('Return Home') }}</a>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                @endif
                            </section>
                        </div>
                    </div>
                </div>
            @endif
            @if (isset($setting->invoice) && $setting->invoice == 'on')
                <div id="tabs-7-1" class="">
                    <div class="card" style="background-color:transparent !important">
                        <div class="card-header"
                            style="padding: 25px 35px !important; background-color:#ffffff !important">
                            <div class="d-flex justify-content-between align-items-center">
                                <div class="row">
                                    <h5 class="mb-0">{{ __('Invoice') }}</h5>

                                </div>
                            </div>
                        </div>
                        <div class="card-body">
                            <section class="section">
                                @if (!empty($project) && !empty($invoices))
                                    <div class="table-responsive">
                                        <table class="table mb-0 ">
                                            <thead>
                                                <tr>
                                                    <th> {{ __('Invoice') }}</th>
                                                    <th>{{ __('Issue Date') }}</th>
                                                    <th>{{ __('Due Date') }}</th>
                                                    <th>{{ __('Due Amount') }}</th>
                                                    <th>{{ __('Status') }}</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                @forelse($invoices as $invoice)
                                                    <tr>
                                                        <td class="Id">
                                                            <a href="{{ route('pay.invoice', \Illuminate\Support\Facades\Crypt::encrypt($invoice->id)) }}"
                                                                target="_new"
                                                                class="btn btn-outline-primary">{{ App\Models\Invoice::invoiceNumberFormat($invoice->invoice_id, $project->created_by, $project->workspace) }}</a>
                                                        </td>
                                                        <td>{{ company_date_formate($invoice->issue_date, $project->created_by, $project->workspace) }}
                                                        </td>
                                                        <td>
                                                            @if ($invoice->due_date < date('Y-m-d'))
                                                                <p class="text-danger">
                                                                    {{ company_date_formate($invoice->due_date, $project->created_by, $project->workspace) }}
                                                                </p>
                                                            @else
                                                                {{ company_date_formate($invoice->due_date, $project->created_by, $project->workspace) }}
                                                            @endif
                                                        </td>
                                                        <td>{{ currency_format_with_sym($invoice->getDue(), $project->created_by, $project->workspace) }}
                                                        </td>
                                                        <td>
                                                            @if ($invoice->status == 0)
                                                                <span
                                                                    class="badge fix_badges bg-info p-2 px-3">{{ __(App\Models\Invoice::$statues[$invoice->status]) }}</span>
                                                            @elseif($invoice->status == 1)
                                                                <span
                                                                    class="badge fix_badges bg-primary p-2 px-3">{{ __(App\Models\Invoice::$statues[$invoice->status]) }}</span>
                                                            @elseif($invoice->status == 2)
                                                                <span
                                                                    class="badge fix_badges bg-secondary p-2 px-3">{{ __(App\Models\Invoice::$statues[$invoice->status]) }}</span>
                                                            @elseif($invoice->status == 3)
                                                                <span
                                                                    class="badge fix_badges bg-warning p-2 px-3">{{ __(App\Models\Invoice::$statues[$invoice->status]) }}</span>
                                                            @elseif($invoice->status == 4)
                                                                <span
                                                                    class="badge fix_badges bg-success p-2 px-3">{{ __(App\Models\Invoice::$statues[$invoice->status]) }}</span>
                                                            @endif
                                                        </td>
                                                    </tr>
                                                @empty
                                                    @include('layouts.nodatafound')
                                                @endforelse
                                            </tbody>
                                        </table>
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
                                                                <a class="btn-return-home badge-blue"
                                                                    href="{{ route('home') }}"><i
                                                                        class="fas fa-reply"></i>
                                                                    {{ __('Return Home') }}</a>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                @endif
                            </section>
                        </div>
                    </div>
                </div>
            @endif

            @if (module_is_active('Account'))
                @if (isset($setting->bill) && $setting->bill == 'on')
                    <div id="tabs-7-1-2" class="">
                        <div class="row">
                            <div class="col-md-12">
                                <div class="card">
                                    <div class="card-header">
                                        <div class="d-flex justify-content-between align-items-center">
                                            <div>
                                                <h5 class="mb-0"> {{ __('Bill') }}</h5>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="card-body table-border-style">
                                        <div class="table-responsive">
                                            <table class="table mb-0 ">
                                                <thead>
                                                    <tr>
                                                        <th> {{ __('Bill') }}</th>

                                                        <th> {{ __('Vendor') }}</th>

                                                        <th> {{ __('Bill Date') }}</th>
                                                        <th> {{ __('Due Date') }}</th>
                                                        <th>{{ __('Due Amount') }}</th>
                                                        <th>{{ __('Status') }}</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    @forelse ($bills as $bill)
                                                        <tr class="font-style">
                                                            <td class="Id">
                                                                @permission('bill show')
                                                                    <a target="_new"
                                                                        href="{{ route('pay.billpay', \Illuminate\Support\Facades\Crypt::encrypt($bill->id)) }}"
                                                                        class="btn btn-outline-primary">{{ Workdo\Account\Entities\Bill::billNumberFormat($bill->bill_id, $bill->created_by, $bill->workspace) }}</a>
                                                                @else
                                                                    <a
                                                                        class="btn btn-outline-primary">{{ Workdo\Account\Entities\Bill::billNumberFormat($bill->bill_id, $bill->created_by, $bill->workspace) }}</a>
                                                        @endif
                                                        </td>


                                                        <td> {{ !empty($bill->vendor) ? $bill->vendor->name : '' }}</td>

                                                        <td>{{ company_date_formate($bill->bill_date, $bill->created_by, $bill->workspace) }}
                                                        </td>
                                                        <td>
                                                            @if ($bill->due_date < date('Y-m-d'))
                                                                <p class="text-danger">
                                                                    {{ company_date_formate($bill->due_date, $bill->created_by, $bill->workspace) }}
                                                                </p>
                                                            @else
                                                                {{ company_date_formate($bill->due_date, $bill->created_by, $bill->workspace) }}
                                                            @endif
                                                        </td>
                                                        <td>{{ currency_format_with_sym($bill->getDue(), $bill->created_by, $bill->workspace) }}
                                                        </td>
                                                        <td>
                                                            @if ($bill->status == 0)
                                                                <span
                                                                    class="badge fix_badges bg-info p-2 px-3 bill_status">{{ __(Workdo\Account\Entities\Bill::$statues[$bill->status]) }}</span>
                                                            @elseif($bill->status == 1)
                                                                <span
                                                                    class="badge fix_badges bg-primary p-2 px-3 bill_status">{{ __(Workdo\Account\Entities\Bill::$statues[$bill->status]) }}</span>
                                                            @elseif($bill->status == 2)
                                                                <span
                                                                    class="badge fix_badges bg-secondary p-2 px-3 bill_status">{{ __(Workdo\Account\Entities\Bill::$statues[$bill->status]) }}</span>
                                                            @elseif($bill->status == 3)
                                                                <span
                                                                    class="badge fix_badges bg-warning p-2 px-3 bill_status">{{ __(Workdo\Account\Entities\Bill::$statues[$bill->status]) }}</span>
                                                            @elseif($bill->status == 4)
                                                                <span
                                                                    class="badge fix_badges bg-success p-2 px-3 bill_status">{{ __(Workdo\Account\Entities\Bill::$statues[$bill->status]) }}</span>
                                                            @endif
                                                        </td>
                                                        </tr>
                                                    @empty
                                                        @include('layouts.nodatafound')
                    @endforelse
                    </tbody>
                    </table>
            </div>
        </div>
        </div>
        </div>
        </div>
        </div>

        </div>
        @endif

        @endif
        @if (module_is_active('Retainer'))
            @if (isset($setting->retainer) && $setting->retainer == 'on')
                <div id="tabs-12" class="">
                    <div class="card" style="background-color:transparent !important">
                        <div class="card-header" style="padding: 25px 35px !important; background-color:#ffffff !important">
                            <div class="d-flex justify-content-between align-items-center">
                                <div class="row">
                                    <h5 class="mb-0">{{ __('Retainer') }}</h5>

                                </div>
                            </div>
                        </div>
                        <div class="card-body">
                            <section class="section">
                                @if (!empty($retainers))
                                    <div class="table-responsive">
                                        <table class="table " id="assets">
                                            <thead>
                                                <tr>
                                                    <th> {{ __('Retainer') }}</th>
                                                    <th> {{ __('Customer') }}</th>
                                                    <th>{{ __('Account Type') }}</th>
                                                    <th> {{ __('Issue Date') }}</th>
                                                    <th>{{ __('Due Amount') }}</th>
                                                    <th> {{ __('Status') }}</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                @foreach ($retainers as $retainer)
                                                    <tr class="font-style">
                                                        <td class="Id">
                                                            <a href="{{ route('pay.retainer', \Crypt::encrypt($retainer->id)) }}"
                                                                class="btn btn-outline-primary">{{ Workdo\Retainer\Entities\Retainer::retainerNumberFormat($retainer->retainer_id, $retainer->created_by, $retainer->workspace) }}
                                                            </a>
                                                        </td>

                                                        <td> {{ !empty($retainer->customer) ? $retainer->customer->name : '' }}
                                                        </td>
                                                        <td>{{ $retainer->account_type }}</td>
                                                        <td>{{ company_date_formate($retainer->issue_date, $retainer->created_by, $retainer->workspace) }}
                                                        </td>
                                                        <td>{{ currency_format_with_sym($retainer->getDue(), $retainer->created_by, $retainer->workspace) }}
                                                        </td>
                                                        <td>
                                                            @if ($retainer->status == 0)
                                                                <span
                                                                    class="badge fix_badges bg-info p-2 px-3">{{ __(Workdo\Retainer\Entities\Retainer::$statues[$retainer->status]) }}</span>
                                                            @elseif($retainer->status == 1)
                                                                <span
                                                                    class="badge fix_badges bg-primary p-2 px-3">{{ __(Workdo\Retainer\Entities\Retainer::$statues[$retainer->status]) }}</span>
                                                            @elseif($retainer->status == 2)
                                                                <span
                                                                    class="badge fix_badges bg-secondary p-2 px-3">{{ __(Workdo\Retainer\Entities\Retainer::$statues[$retainer->status]) }}</span>
                                                            @elseif($retainer->status == 3)
                                                                <span
                                                                    class="badge fix_badges bg-danger p-2 px-3">{{ __(Workdo\Retainer\Entities\Retainer::$statues[$retainer->status]) }}</span>
                                                            @elseif($retainer->status == 4)
                                                                <span
                                                                    class="badge fix_badges bg-warning p-2 px-3">{{ __(Workdo\Retainer\Entities\Retainer::$statues[$retainer->status]) }}</span>
                                                            @elseif($retainer->status == 5)
                                                                <span
                                                                    class="badge fix_badges bg-success p-2 px-3">{{ __(Workdo\Retainer\Entities\Retainer::$statues[$retainer->status]) }}</span>
                                                            @endif
                                                        </td>
                                                    </tr>
                                                @endforeach
                                            </tbody>
                                        </table>
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
                                                                <a class="btn-return-home badge-blue"
                                                                    href="{{ route('home') }}"><i class="fas fa-reply"></i>
                                                                    {{ __('Return Home') }}</a>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                @endif
                            </section>
                        </div>
                    </div>
                </div>
            @endif
        @endif
        @if (isset($setting->proposal) && $setting->proposal == 'on')
            <div id="tabs-13" class="">
                <div class="card" style="background-color:transparent !important">
                    <div class="card-header" style="padding: 25px 35px !important; background-color:#ffffff !important">
                        <div class="d-flex justify-content-between align-items-center">
                            <div class="row">
                                <h5 class="mb-0">{{ __('proposals') }}</h5>

                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <section class="section">
                            @if (!empty($proposals))
                                <div class="table-responsive">
                                    <table class="table " id="assets">
                                        <thead>
                                            <tr>
                                                <th> {{ __('Proposal') }}</th>
                                                <th> {{ __('Customer') }}</th>
                                                <th>{{ __('Account Type') }}</th>
                                                <th> {{ __('Issue Date') }}</th>
                                                <th> {{ __('Status') }}</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            @foreach ($proposals as $proposal)
                                                <tr class="font-style">
                                                    <td class="Id">
                                                        <a href="{{ route('pay.proposalpay', \Crypt::encrypt($proposal->id)) }}"
                                                            class="btn btn-outline-primary">{{ \App\Models\Proposal::proposalNumberFormat($proposal->proposal_id, $proposal->created_by, $proposal->workspace) }}
                                                        </a>
                                                    </td>
                                                    <td> {{ !empty($proposal->customer) ? $proposal->customer->name : '' }}
                                                    </td>
                                                    <td>{{ $proposal->account_type }}</td>
                                                    <td>{{ company_date_formate($proposal->issue_date, $proposal->created_by, $proposal->workspace) }}
                                                    </td>
                                                    <td>
                                                        @if ($proposal->status == 0)
                                                            <span
                                                                class="badge fix_badge bg-primary p-2 px-3">{{ __(\App\Models\Proposal::$statues[$proposal->status]) }}</span>
                                                        @elseif($proposal->status == 1)
                                                            <span
                                                                class="badge fix_badge bg-info p-2 px-3">{{ __(\App\Models\Proposal::$statues[$proposal->status]) }}</span>
                                                        @elseif($proposal->status == 2)
                                                            <span
                                                                class="badge fix_badge bg-secondary p-2 px-3">{{ __(\App\Models\Proposal::$statues[$proposal->status]) }}</span>
                                                        @elseif($proposal->status == 3)
                                                            <span
                                                                class="badge fix_badge bg-warning p-2 px-3">{{ __(\App\Models\Proposal::$statues[$proposal->status]) }}</span>
                                                        @elseif($proposal->status == 4)
                                                            <span
                                                                class="badge fix_badge bg-danger p-2 px-3">{{ __(\App\Models\Proposal::$statues[$proposal->status]) }}</span>
                                                        @endif
                                                    </td>
                                                </tr>
                                            @endforeach
                                        </tbody>
                                    </table>
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
                                                            <a class="btn-return-home badge-blue"
                                                                href="{{ route('home') }}"><i class="fas fa-reply"></i>
                                                                {{ __('Return Home') }}</a>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            @endif
                        </section>
                    </div>
                </div>
            </div>
        @endif
        @if (module_is_active('Timesheet', $project->created_by))
            @if (isset($setting->timesheet) && $setting->timesheet == 'on')
                <div id="tabs-8" class="">
                    <div class="row">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <div>
                                            <h5 class="mb-0"> {{ __('Timesheet') }}</h5>
                                        </div>
                                    </div>
                                </div>
                                <div class="card-body">
                                    @include('timesheet::project.index', ['project' => $project])
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            @endif
        @endif

        @if (isset($setting->activity) && $setting->activity == 'on')
            <div id="tabs-9" class="">
                <div class="card">
                    <div class="card-header">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <h5 class="mb-0">{{ __('Activity') }}</h5>
                            </div>
                        </div>
                    </div>
                    <div class="card-body p-3">
                        <div class="timeline timeline-one-side top-10-scroll" data-timeline-content="axis"
                            data-timeline-axis-style="dashed" style="max-height: 300px;">
                            @foreach ($project->activities as $activity)
                                <div class="timeline-block px-2 pt-3">
                                    @if ($activity->log_type == 'Upload File')
                                        <span class="timeline-step timeline-step-sm border border-primary text-white"> <i
                                                class="fas fa-file text-dark"></i></span>
                                    @elseif($activity->log_type == 'Create Milestone')
                                        <span class="timeline-step timeline-step-sm border border-info text-white"> <i
                                                class="fas fa-cubes text-dark"></i></span>
                                    @elseif($activity->log_type == 'Create Task')
                                        <span class="timeline-step timeline-step-sm border border-success text-white"> <i
                                                class="fas fa-tasks text-dark"></i></span>
                                    @elseif($activity->log_type == 'Create Bug')
                                        <span class="timeline-step timeline-step-sm border border-warning text-white"> <i
                                                class="fas fa-bug text-dark"></i></span>
                                    @elseif($activity->log_type == 'Move' || $activity->log_type == 'Move Bug')
                                        <span class="timeline-step timeline-step-sm border round border-danger text-white"> <i
                                                class="fas fa-align-justify text-dark"></i></span>
                                    @elseif($activity->log_type == 'Create Invoice')
                                        <span class="timeline-step timeline-step-sm border border-bg-dark text-white"> <i
                                                class="fas fa-file-invoice text-dark"></i></span>
                                    @elseif($activity->log_type == 'Invite User')
                                        <span class="timeline-step timeline-step-sm border border-success"> <i
                                                class="fas fa-plus text-dark"></i></span>
                                    @elseif($activity->log_type == 'Share with Client')
                                        <span class="timeline-step timeline-step-sm border border-info text-white"> <i
                                                class="fas fa-share text-dark"></i></span>
                                    @elseif($activity->log_type == 'Create Timesheet')
                                        <span class="timeline-step timeline-step-sm border border-success text-white"> <i
                                                class="fas fa-clock-o text-dark"></i></span>
                                    @endif
                                    <div class="row last_notification_text">
                                        <span class="col-1 m-0 h6 text-sm"> <span>{{ $activity->log_type }} </span> </span>
                                        <br>
                                        <span class="col text-sm h6"> {!! $activity->getRemark() !!} </span>
                                        <div class="col-auto notification_time_main">
                                            <p class="text-muted">{{ $activity->created_at->diffForHumans() }}</p>
                                        </div>
                                    </div>

                                </div>
                            @endforeach
                        </div>
                    </div>
                </div>
            </div>
        @endif

        @if (isset($setting->documents) && $setting->documents == 'on')
            @stack('DocumentShareSection')
        @endif
        @if (isset($setting->procurement) && $setting->procurement == 'on')
            <div id="tab_14">
                @stack('ProcurementShareSection')
            </div>
        @endif


        </div>
        </div>
    @endsection
    @push('scripts')
        <script src="{{ asset('assets/js/plugins/apexcharts.min.js') }}"></script>
        <script src="{{ asset('js/letter.avatar.js') }}"></script>
        <script>
            (function() {
                var options = {
                    chart: {
                        height: 250,
                        type: 'line',
                        toolbar: {
                            show: false,
                        },
                    },
                    dataLabels: {
                        enabled: false
                    },
                    stroke: {
                        width: 2,
                        curve: 'smooth'
                    },
                    series: [
                        @foreach ($chartData['stages'] as $id => $name)
                            {
                                name: "{{ __($name) }}",
                                // data:
                                data: {!! json_encode($chartData[$id]) !!},
                            },
                        @endforeach
                    ],
                    xaxis: {
                        categories: {!! json_encode($chartData['label']) !!},
                    },
                    colors: {!! json_encode($chartData['color']) !!},

                    grid: {
                        strokeDashArray: 4,
                    },
                    legend: {
                        show: false,
                    },

                    yaxis: {
                        tickAmount: 5,
                        min: 1,
                        max: 40,
                    },
                };
                var chart = new ApexCharts(document.querySelector("#task-chart"), options);
                chart.render();
            })();
            if ($('#useradd-sidenav').length > 0) {
                var scrollSpy = new bootstrap.ScrollSpy(document.body, {
                    target: '#useradd-sidenav',
                    offset: 300,
                });
                $(".list-group-item").click(function() {
                    $('.list-group-item').filter(function() {
                        return this.href == id;
                    }).parent().removeClass('text-primary');
                });
            }
        </script>
    @endpush
