@extends('layouts.main')
@php
    if ($project->type == 'project') {
        $name = 'Project';
    } else {
        $name = 'Project Template';
    }
@endphp
@section('page-title')
    {{ __($name . ' Detail') }}
@endsection
@push('css')
    <link rel="stylesheet" href="{{ asset('assets/css/plugins/dropzone.css') }}" type="text/css" />
    <link rel="stylesheet" href="{{ asset('packages/workdo/Taskly/src/Resources/assets/css/custom.css') }}" type="text/css" />
@endpush
@section('page-breadcrumb')
    {{ __($name . ' Detail') }}
@endsection
@section('page-action')

<div class="d-flex">
        @if ($project->type == 'project')
            @stack('addButtonHook')
        @else
            @stack('projectConvertButton')
        @endif
    <div class="col-md-auto  pb-3">
        <a href="#" class="btn btn-sm  align-items-center cp_link bg-primary me-2"
            data-link="{{ route('project.shared.link', [\Illuminate\Support\Facades\Crypt::encrypt($project->id)]) }}"
            data-toggle="tooltip" data-bs-toggle="tooltip" data-bs-original-title="{{ __('Copy') }}">
            <span class="btn-inner--text text-white">
                <i class="ti ti-copy"></i></span>
        </a>
    </div>
    @permission('project setting')
        @php
            $title =
                module_is_active('ProjectTemplate') && $project->type == 'template'
                    ? __('Shared Project Template Settings')
                    : __('Shared Project Settings');
        @endphp
        <div class="col-sm-auto">
            <a href="#" class="btn btn-sm me-2 btn-primary" data-title="{{ $title }}"
                data-ajax-popup="true" data-size="md" data-bs-toggle="tooltip"
                data-bs-original-title="{{ __('Shared Project Setting') }}"
                data-url="{{ route('project.setting', [$project->id]) }}">
                <i class="ti ti-settings"></i>
            </a>
        </div>
    @endpermission
    @permission('task manage')
        <div class="col-sm-auto">
            <a class="btn btn-sm me-2 btn-primary" data-bs-toggle="tooltip" href="{{ route('projects.calendar',[$project->id]) }}"
                data-bs-original-title="{{ __('Calendar') }}">
                <i class="ti ti-calendar"></i>
            </a>
        </div>
    @endpermission
        <div class="col-sm-auto">
            <a class="btn btn-sm me-2 btn-primary" data-bs-toggle="tooltip" href="{{ route('projects.gantt', [$project->id]) }}"
                data-bs-original-title="{{ __('Gantt Chart') }}">
                <i class="ti ti-chart-bar"></i>
            </a>
        </div>
    {{-- @permission('project tracker manage') --}}
        {{-- @if (module_is_active('TimeTracker'))
            <div class="col-sm-auto">
                <a href="{{ route('projecttime.tracker', [$project->id]) }}"
                    class="btn btn-xs btn-primary btn-icon-only width-auto ">{{ __('Tracker') }}</a>
            </div>
        @endif --}}
    {{-- @endpermission --}}
    @permission('project finance manage')
        <div class="col-sm-auto">
            <a class="btn btn-sm me-2 btn-primary" data-bs-toggle="tooltip" href="{{ route('projects.proposal', [$project->id]) }}"
                data-bs-original-title="{{ __('Finance') }}">
                <i class="ti ti-file-analytics"></i>
            </a>
        </div>
    @endpermission
    @if (module_is_active('Procurement'))
        <div class="col-sm-auto">
            <a class="btn btn-sm me-2 btn-primary" data-bs-toggle="tooltip" href="{{ route('rfx.index') }}"
                data-bs-original-title="{{ __('RFx') }}">
                <i class="ti ti-clipboard"></i>
            </a>
        </div>
    @endif
    @permission('bug manage')
        <div class="col-sm-auto">
            <a class="btn btn-sm me-2 btn-primary" data-bs-toggle="tooltip" href="{{ route('projects.bug.report', [$project->id]) }}"
                data-bs-original-title="{{ __('Bug Report') }}">
                <i class="ti ti-bug"></i>
            </a>
        </div>
    @endpermission
    @permission('task manage')
        <div class="col-sm-auto">
            <a class="btn btn-sm btn-primary" data-bs-toggle="tooltip" href="{{ route('projects.task.board', [$project->id]) }}"
                data-bs-original-title="{{ __('Task Board') }}">
                <i class="ti ti-layout-kanban"></i>
            </a>
        </div>
    @endpermission
</div>

@endsection
@push('css')
    <link rel="stylesheet" href="{{ asset('assets/css/plugins/dropzone.min.css') }}">
    <style>
        @media (max-width: 1300px) {
            .row1 {
                display: flex;
                flex-wrap: wrap;
            }
        }
    </style>
@endpush
@section('content')
    <div class="row row-gap mb-4 ">
        <div class="col-xxl-6 col-12">
            <div class="dashboard-card project-detail-card">
                <img src="{{ asset('assets/images/layer.png')}}" class="dashboard-card-layer" alt="layer">
                <div class="card-inner">
                    <div class="card-content">
                        <h2 class="text-primary">{{ $project->name }}</h2>
                        <p>{{ $project->description }}</p>
                        @if (!empty($customFields) && count($project->customField) > 0)
                            <div class="pt-3">
                                @foreach ($customFields as $field)
                                    <div class="mb-2">
                                        <p>
                                            <strong class="text-light">{{ $field->name }} :</strong>
                                            @if ($field->type == 'attachment')
                                                <a href="{{ get_file($project->customField[$field->id]) }}" target="_blank" class="text-primary">
                                                    <i class="ti ti-file"></i> {{ __('View File') }}
                                                </a>
                                            @else
                                                <span class="text-muted">
                                                    {{ !empty($project->customField[$field->id]) ? $project->customField[$field->id] : '-' }}
                                                </span>
                                            @endif
                                        </p>
                                    </div>
                                @endforeach
                            </div>
                        @endif
                        <div class="btn-wrp d-flex gap-3">
                            @permission('project edit')
                                <a href="#" class="btn btn-light" tabindex="0" data-size="lg" data-url="{{ route('projects.edit',$project->id) }}" data-ajax-popup="true" data-title="{{ __('Edit ') . $name }}" data-bs-toggle="tooltip"  title="{{__('Edit')}}" data-original-title="{{__('Edit')}}">
                                    <i class="ti ti-pencil text-success"></i>
                                </a>
                            @endpermission
                            @permission('project delete')
                                {{ Form::open(['route' => ['projects.destroy', $project->id], 'class' => 'm-0']) }}
                                    @method('DELETE')
                                    <a href="#" class="btn btn-light show_confirm" tabindex="0" data-bs-toggle="tooltip" title=""
                                    data-bs-original-title="{{__('Delete')}}" aria-label="{{__('Delete')}}"
                                    data-confirm-yes="delete-form-{{ $project->id }}">
                                        <i class="ti ti-trash text-danger"></i>
                                    </a>
                                {{ Form::close() }}
                            @endpermission
                        </div>
                    </div>
                    @if ($project->type == 'project')
                        <div class="status-info">
                            <div class="status-wrp">
                                <span class="d-block">{{ __('Start Date') }}:</span>
                                <p class="mb-0 text-primary">{{ company_date_formate($project->start_date) }}</p>
                            </div>
                            <div class="status-wrp">
                                <span class="d-block">{{ __('Due Date') }}:</span>
                                <p class="mb-0 text-primary">{{ company_date_formate($project->end_date) }}</p>
                            </div>
                            <div class="status-wrp">
                                <span class="d-block">{{ __('Total Members') }}:</span>
                                <p class="mb-0 text-primary">{{ (int) $project->users->count() + (int) $project->clients->count() }}</p>
                            </div>
                            <div class="status-wrp ">
                                <span class="d-block">
                                    @if ($project->status == 'Finished')
                                        <div class="badge bg-success p-2 f-12 text-capitalize"> {{ __('Finished') }}
                                        </div>
                                    @elseif($project->status == 'Ongoing')
                                        <div class="badge bg-secondary p-2 f-12 text-capitalize">{{ __('Ongoing') }}
                                        </div>
                                    @else
                                        <div class="badge bg-warning p-2 f-12 text-capitalize">{{ __('OnHold') }}
                                        </div>
                                    @endif
                                </span>
                            </div>
                        </div>
                    @endif
                </div>
            </div>
        </div>
        <div class="col-xxl-6 col-12">
            <div class="row dashboard-wrp">
                 <div class="col-sm-6 col-12">
                    <div class="dashboard-project-card">
                        <div class="card-inner  d-flex justify-content-between">
                            <div class="card-content">
                                <div class="theme-avtar bg-white">
                                    <i class="fas fas fa-calendar-day text-danger"></i>
                                </div>
                                <h3 class="mt-3 mb-0 text-danger">{{ __('Total Tasks') }}</h3>
                            </div>
                            <h3 class="mb-0">{{ $project->countTask() }}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-sm-6 col-12">
                    <div class="dashboard-project-card">
                        <div class="card-inner d-flex justify-content-between">
                            <div class="card-content">
                                <div class="theme-avtar bg-white">
                                    <i class="ti ti-message-circle-2"></i>
                                </div>
                                <h3 class="mt-3 mb-0">{{ __('Comment') }}</h3>
                            </div>
                            <h3 class="mb-0">{{ $project->countTaskComments() }}</h3>
                        </div>
                    </div>
                </div>
                @if ($project->type == 'project')
                    <div class="col-sm-6 col-12">
                        <div class="dashboard-project-card">
                            <div class="card-inner  d-flex justify-content-between">
                                <div class="card-content">
                                    <div class="theme-avtar bg-white">
                                        <i class="fas fas fa-calendar-day"></i>
                                    </div>
                                    <h3 class="mt-3 mb-0">{{ __('Days left') }}</h3>
                                </div>
                                <h3 class="mb-0">{{ $daysleft }}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-sm-6 col-12">
                        <div class="dashboard-project-card">
                            <div class="card-inner  d-flex justify-content-between">
                                <div class="card-content">
                                    <div class="theme-avtar bg-white">
                                        <i class="fas fa-money-bill-alt"></i>
                                    </div>
                                    <h3 class="mt-3 mb-0">{{ __('Budget') }}</h3>
                                </div>
                                <h3 class="mb-0">{{ company_setting('defult_currancy') }}
                                    {{ number_format($project->budget) }}</h3>
                            </div>
                        </div>
                    </div>
                @endif
            </div>
        </div>
    </div>
    <div class="row project-detail-wrp">
        <!-- [ sample-page ] start -->
                @if ($project->type == 'project')
                    <div class="col-xxl-4 col-md-6">
                        <div class="card deta-card">
                            <div class="card-header p-3">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <h5 class="mb-0">{{ __('Team Members') }}
                                            ({{ count($project->users) }})
                                        </h5>
                                    </div>
                                    <div class="text-end">
                                        <p class="text-muted d-sm-flex align-items-center mb-0">

                                            <a href="#" class="btn btn-sm btn-primary"
                                                data-ajax-popup="true" data-title="{{ __('Invite') }}"
                                                data-bs-toggle="tooltip" data-bs-title="{{ __('Invite') }}"
                                                data-url="{{ route('projects.invite.popup', [$project->id]) }}"><i
                                                    class="ti ti-brand-telegram"></i></a>
                                        </p>
                                    </div>
                                </div>
                            </div>
                            <div class="card-body p-3 top-10-scroll">
                                @foreach ($project->users as $user)
                                    <ul class="list-group list-group-flush mb-3" style="width: 100%;">
                                        <li class="list-group-item p-3">
                                            <div class="row align-items-center justify-content-between">
                                                <div class="col-sm-auto mb-3 mb-sm-0">
                                                    <div class="d-flex align-items-center justify-content-center">
                                                        <a href="#" class=" text-start">
                                                            <img alt="image" data-bs-toggle="tooltip"
                                                                data-bs-placement="top"
                                                                title="{{ $user->name }}"
                                                                @if ($user->avatar) src="{{ get_file($user->avatar) }}" @else src="{{ get_file('avatar.png') }}" @endif
                                                                class="rounded  border-1 border border-primary" width="40px"
                                                                height="40px">
                                                        </a>
                                                        <div class="px-2">
                                                            <h5 class="m-0">{{ $user->name }}</h5>
                                                            <small class="text-muted">{{ $user->email }}<span
                                                                    class="text-primary "> -
                                                                    {{ (int) count($project->user_done_tasks($user->id)) }}/{{ (int) count($project->user_tasks($user->id)) }}</span></small>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div class="col-sm-auto text-sm-end d-flex align-items-center justify-content-center">
                                                    @auth('web')
                                                        @if ($user->id != Auth::id())
                                                            @permission('team member remove')
                                                                <form id="delete-user-{{ $user->id }}"
                                                                    action="{{ route('projects.user.delete', [$project->id, $user->id]) }}"
                                                                    method="POST" style="display: none;"
                                                                    class="d-inline-flex">
                                                                    <a href="#"
                                                                        class="btn btn-sm align-items-center text-white show_confirm bg-danger"
                                                                        data-confirm="{{ __('Are You Sure?') }}"
                                                                        data-text="{{ __('This action can not be undone. Do you want to continue?') }}"
                                                                        data-confirm-yes="delete-user-{{ $user->id }}"
                                                                        data-toggle="tooltip"
                                                                        title="{{ __('Delete') }}"><i
                                                                            class="ti ti-trash"></i></a>

                                                                    @csrf
                                                                    @method('DELETE')
                                                                </form>
                                                            @endpermission
                                                        @endif
                                                    @endauth
                                                </div>
                                            </div>
                                        </li>
                                    </ul>
                                @endforeach
                            </div>
                        </div>
                    </div>
                    <div class="col-xxl-4 col-md-6">
                        <div class="card deta-card">
                            <div class="card-header p-3">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <h5 class="mb-0">{{ __('Clients') }} ({{ count($project->clients) }})
                                        </h5>
                                    </div>
                                    <div class="text-end">
                                        <p class="text-muted d-none d-sm-flex align-items-center mb-0">
                                            <a href="#" class="btn btn-sm btn-primary"
                                                data-ajax-popup="true" data-title="{{ __('Share to Client') }}"
                                                data-toggle="tooltip" title="{{ __('Share to Client') }}"
                                                data-url="{{ route('projects.share.popup', [$project->id]) }}"><i
                                                    class="ti ti-share"></i></a>
                                        </p>
                                    </div>
                                </div>
                            </div>
                            <div class="card-body p-3 top-10-scroll">
                                @foreach ($project->clients as $client)
                                    <ul class="list-group list-group-flush mb-3" style="width: 100%;">
                                        <li class="list-group-item p-3">
                                            <div class="row align-items-center justify-content-between">
                                                <div class="col-sm-auto mb-3 mb-sm-0">
                                                    <div class="d-flex align-items-center justify-content-center">
                                                        <a href="#" class=" text-start">
                                                            <img alt="image" data-bs-toggle="tooltip"
                                                                data-bs-placement="top"
                                                                title="{{ $client->name }}"
                                                                @if ($client->avatar) src="{{ get_file($client->avatar) }}" @else src="{{ get_file('avatar.png') }}" @endif
                                                                class="rounded border-1 border border-primary" width="40px"
                                                                height="40px">
                                                        </a>
                                                        <div class="px-2">
                                                            <h5 class="m-0">{{ $client->name }}</h5>
                                                            <small class="text-muted">{{ $client->email }}</small>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div class="col-sm-auto text-sm-end d-flex align-items-center justify-content-center">
                                                    @if (\Auth::user()->hasRole('company'))
                                                        @permission('team client remove')
                                                            <form id="delete-client-{{ $client->id }}"
                                                                action="{{ route('projects.client.delete', [$project->id, $client->id]) }}"
                                                                method="POST" style="display: none;"
                                                                class="d-inline-flex">
                                                                <a href="#"
                                                                    class="btn btn-sm align-items-center text-white show_confirm bg-danger"
                                                                    data-confirm="{{ __('Are You Sure?') }}"
                                                                    data-text="{{ __('This action can not be undone. Do you want to continue?') }}"
                                                                    data-confirm-yes="delete-client-{{ $client->id }}"
                                                                    data-toggle="tooltip"
                                                                    title="{{ __('Delete') }}"><i
                                                                        class="ti ti-trash"></i></a>
                                                                @csrf
                                                                @method('DELETE')

                                                            </form>
                                                        @endpermission
                                                    @endif
                                                </div>
                                            </div>
                                        </li>
                                    </ul>
                                @endforeach
                            </div>
                        </div>
                    </div>
                    @if (module_is_active('Account', $project->created_by))
                        <div class="col-xxl-4 col-md-6">
                            <div class="card deta-card">
                                <div class="card-header p-3">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <div>
                                            <h5 class="mb-0">{{ __('Vendors') }}
                                                ({{ count($project->venders) }})</h5>
                                        </div>
                                        <div class="text-end">
                                            <p class="text-muted d-none d-sm-flex align-items-center mb-0">
                                                <a href="#" class="btn btn-sm btn-primary"
                                                    data-ajax-popup="true"
                                                    data-title="{{ __('Share to vendor') }}"
                                                    data-toggle="tooltip" title="{{ __('Share to vendor') }}"
                                                    data-url="{{ route('projects.share.vender.popup', [$project->id]) }}"><i
                                                    class="ti ti-share"></i>
                                                </a>
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                <div class="card-body p-3 top-10-scroll">
                                    @foreach ($project->venders as $client)
                                        <ul class="list-group list-group-flush mb-3" style="width: 100%;">
                                            <li class="list-group-item p-3">
                                                <div class="row align-items-center justify-content-between">
                                                    <div class="col-sm-auto mb-3 mb-sm-0">
                                                        <div class="d-flex align-items-center justify-content-center">
                                                            <a href="#" class=" text-start">
                                                                <img alt="image" data-bs-toggle="tooltip"
                                                                    data-bs-placement="top"
                                                                    title="{{ $client->name }}"
                                                                    @if ($client->avatar) src="{{ get_file($client->avatar) }}" @else src="{{ get_file('avatar.png') }}" @endif
                                                                    class="rounded border-1 border border-primary" width="40px"
                                                                    height="40px">
                                                            </a>
                                                            <div class="px-2">
                                                                <h5 class="m-0">{{ $client->name }}</h5>
                                                                <small class="text-muted">{{ $client->email }}</small>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div class="col-sm-auto text-sm-end d-flex align-items-center justify-content-center">
                                                        @if (\Auth::user()->hasRole('company'))
                                                            @permission('team client remove')
                                                                <form id="delete-client-{{ $client->id }}"
                                                                    action="{{ route('projects.vendor.delete', [$project->id, $client->id]) }}"
                                                                    method="POST" style="display: none;"
                                                                    class="d-inline-flex">
                                                                    <a href="#"
                                                                        class="btn btn-sm align-items-center text-white show_confirm bg-danger"
                                                                        data-confirm="{{ __('Are You Sure?') }}"
                                                                        data-text="{{ __('This action can not be undone. Do you want to continue?') }}"
                                                                        data-confirm-yes="delete-client-{{ $client->id }}"
                                                                        data-toggle="tooltip"
                                                                        title="{{ __('Delete') }}"><i
                                                                        class="ti ti-trash"></i></a>
                                                                    @csrf
                                                                    @method('DELETE')
                                                                </form>
                                                            @endpermission
                                                        @endif
                                                    </div>
                                                </div>
                                            </li>
                                        </ul>
                                    @endforeach
                                </div>
                            </div>
                        </div>
                    @else
                        <div class="col-12">
                            <div class="card deta-card">
                                <div class="card-header p-3">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <div>
                                            <h5 class="mb-0">{{ __('Activity') }}</h5>
                                        </div>
                                    </div>
                                </div>
                                <div class="card-body top-10-scroll p-3" style="max-height: 300px;">
                                    <div class="timeline timeline-one-side  row row-gap">
                                        @foreach ($project->activities as $activity)
                                            <div class="col-sm-6 col-12">
                                                <div class="timeline-card">
                                                    <div class="timeline-content-top d-flex gap-3">
                                                        @if ($activity->log_type == 'Upload File')
                                                            <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                                <i class="fas fa-file text-primary"></i>
                                                            </div>
                                                        @elseif($activity->log_type == 'Create Milestone')
                                                            <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                                <i class="fas fa-cubes text-primary"></i>
                                                            </div>
                                                        @elseif($activity->log_type == 'Create Task')
                                                            <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                                <i class="fas fa-tasks text-primary"></i>
                                                            </div>
                                                        @elseif($activity->log_type == 'Create Bug')
                                                            <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                                <i class="fas fa-bug text-primary"></i>
                                                            </div>
                                                        @elseif($activity->log_type == 'Move' || $activity->log_type == 'Move Bug')
                                                            <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                                <i class="fas fa-align-justify text-primary"></i>
                                                            </div>
                                                        @elseif($activity->log_type == 'Create Invoice')
                                                            <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                                <i class="fas fa-file-invoice text-primary"></i>
                                                            </div>
                                                        @elseif($activity->log_type == 'Invite User')
                                                            <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                                <i class="fas fa-plus text-primary"></i>
                                                            </div>
                                                        @elseif($activity->log_type == 'Share with Client' || $activity->log_type == 'Share with Vender')
                                                            <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                                <i class="fas fa-share text-primary"></i>
                                                            </div>
                                                        @elseif($activity->log_type == 'Create Timesheet')
                                                            <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                                <i class="fas fa-clock-o text-primary"></i>
                                                            </div>
                                                        @endif
                                                        <div class="notification_text">
                                                            <h3 class="h5 mb-1">{{$activity->log_type}}</h3>
                                                            <p class="mb-0">{!! $activity->getRemark() !!}</p>
                                                        </div>
                                                    </div>
                                                    <div class="timeline-content-bottom">
                                                        <p class="mb-0">
                                                            <i class="ti ti-clock"></i>
                                                            {{ $activity->created_at->diffForHumans() }}</p>
                                                    </div>
                                                </div>
                                            </div>
                                        @endforeach
                                    </div>
                                </div>
                            </div>
                        </div>
                    @endif
                @endif
                <div class="col-xxl-8 col-12">
                    <div class="card milestone-card">
                        <div class="card-header p-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h5 class="mb-0">{{ __('Milestones') }} ({{ count($project->milestones) }})
                                    </h5>
                                </div>
                                <div class="text-end">
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
                        <div class="card-body p-3 top-10-scroll">
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
                                            <th>{{ __('Action') }}</th>
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
                                                <td>{{ currency_format_with_sym($milestone->cost) }}
                                                </td>
                                                <td>
                                                    <div class="progress_wrapper">
                                                        <div class="progress" style="width: 100px">
                                                            <div class="progress-bar" role="progressbar"
                                                                style="width: {{ $milestone->progress }}px;"
                                                                aria-valuenow="55" aria-valuemin="0"
                                                                aria-valuemax="100">
                                                            </div>
                                                        </div>
                                                        <div class="progress_labels">
                                                            <div class="total_progress">
                                                                <strong> {{ $milestone->progress }}%</strong>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </td>
                                                <td class="col-auto d-flex">
                                                    @permission('milestone edit')
                                                        <div class="action-btn btn-primary me-2">
                                                            <a class="btn btn-sm align-items-center text-white bg-info mt-1"
                                                                data-ajax-popup="true" data-size="lg"
                                                                data-title="{{ __('Edit Milestone') }}"
                                                                data-url="{{ route('projects.milestone.edit', [$milestone->id]) }}"
                                                                data-toggle="tooltip" title="{{ __('Edit') }}"><i
                                                                class="ti ti-pencil text-white"></i></a>
                                                        </div>
                                                    @endpermission
                                                    @permission('milestone delete')
                                                        <form id="delete-form1-{{ $milestone->id }}"
                                                            action="{{ route('projects.milestone.destroy', [$milestone->id]) }}"
                                                            method="POST" style="display: none;"
                                                            class="d-inline-flex">
                                                            <a href="#"
                                                                class="btn btn-sm align-items-center text-white show_confirm bg-danger"
                                                                data-confirm="{{ __('Are You Sure?') }}"
                                                                data-text="{{ __('This action can not be undone. Do you want to continue?') }}"
                                                                data-confirm-yes="delete-form1-{{ $milestone->id }}"
                                                                data-toggle="tooltip" title="{{ __('Delete') }}"><i
                                                                class="ti ti-trash"></i></a>
                                                            @csrf
                                                            @method('DELETE')
                                                        </form>
                                                    @endpermission
                                                </td>
                                            </tr>
                                        @endforeach
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-xxl-4 col-12">
                    <div class="card" style="height:292px">
                        <div class="card-header p-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h5 class="mb-0">{{ __('Progress') }}<span class="text-end">
                                        ({{ __('Last Week Tasks') }}) </span></h5>
                                </div>
                            </div>
                        </div>
                        <div class="card-body p-2">
                            <div id="task-chart"></div>
                        </div>
                    </div>
                </div>
                @if (module_is_active('Account', $project->created_by) && $project->type == 'project')
                    <div class="col-xl-8 col-12">
                        <div class="card deta-card">
                            <div class="card-header p-3">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <h5 class="mb-0">{{ __('Activity') }}</h5>
                                    </div>
                                </div>
                            </div>
                            <div class="card-body top-10-scroll p-3" style="max-height: 300px;">
                                <div class="timeline timeline-one-side  row row-gap">
                                    @foreach ($project->activities as $activity)
                                        <div class="col-sm-6 col-12">
                                            <div class="timeline-card">
                                                <div class="timeline-content-top d-flex gap-3">
                                                    @if ($activity->log_type == 'Upload File')
                                                        <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                            <i class="fas fa-file text-primary"></i>
                                                        </div>
                                                    @elseif($activity->log_type == 'Create Milestone')
                                                        <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                            <i class="fas fa-cubes text-primary"></i>
                                                        </div>
                                                    @elseif($activity->log_type == 'Create Task')
                                                        <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                            <i class="fas fa-tasks text-primary"></i>
                                                        </div>
                                                    @elseif($activity->log_type == 'Create Bug')
                                                        <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                            <i class="fas fa-bug text-primary"></i>
                                                        </div>
                                                    @elseif($activity->log_type == 'Move' || $activity->log_type == 'Move Bug')
                                                        <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                            <i class="fas fa-align-justify text-primary"></i>
                                                        </div>
                                                    @elseif($activity->log_type == 'Create Invoice')
                                                        <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                            <i class="fas fa-file-invoice text-primary"></i>
                                                        </div>
                                                    @elseif($activity->log_type == 'Invite User')
                                                        <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                            <i class="fas fa-plus text-primary"></i>
                                                        </div>
                                                    @elseif($activity->log_type == 'Share with Client' || $activity->log_type == 'Share with Vender')
                                                        <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                            <i class="fas fa-share text-primary"></i>
                                                        </div>
                                                    @elseif($activity->log_type == 'Create Timesheet')
                                                        <div class="timeline-step timeline-step-sm rounded  border border-primary">
                                                            <i class="fas fa-clock-o text-primary"></i>
                                                        </div>
                                                    @endif
                                                    <div class="notification_text">
                                                        <h3 class="h5 mb-1">{{$activity->log_type}}</h3>
                                                        <p class="mb-0">{!! $activity->getRemark() !!}</p>
                                                    </div>
                                                </div>
                                                <div class="timeline-content-bottom">
                                                    <p class="mb-0">
                                                        <i class="ti ti-clock"></i>
                                                        {{ $activity->created_at->diffForHumans() }}</p>
                                                </div>
                                            </div>
                                        </div>
                                    @endforeach
                                </div>
                            </div>
                        </div>
                    </div>
                @endif
                @if($project->type == 'template')
                    <div class="col-xl-12 col-12">
                        <div class="card ">
                        <div class="card-header p-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h5 class="mb-0">{{ __('Files') }}</h5>
                                </div>
                            </div>
                        </div>
                        <div class="card-body d-flex align-items-center justify-content-center  p-3"  >

                            <div class="author-box-name form-control-label mb-4">

                            </div>
                            <div class="col-12 dropzone browse-file d-flex flex-wrap align-items-center justify-content-center top-10-scroll" id="dropzonewidget" style="height: 260px;">
                                <div class="dz-message" data-dz-message>
                                    <span>{{ __('Drop files here to upload') }}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                @else
                <div class="col-xl-4 col-12">
                    <div class="card ">
                        <div class="card-header p-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h5 class="mb-0">{{ __('Files') }}</h5>
                                </div>
                            </div>
                        </div>
                        <div class="card-body d-flex align-items-center justify-content-center  p-3"  >

                            <div class="author-box-name form-control-label mb-4">

                            </div>
                            <div class="col-12 dropzone browse-file d-flex flex-wrap align-items-center justify-content-center top-10-scroll" id="dropzonewidget" style="height: 260px;">
                                <div class="dz-message" data-dz-message>
                                    <span>{{ __('Drop files here to upload') }}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                @endif
                <div class="col-md-12">
                    @stack('DocumentSection')
                </div>
            <!-- [ sample-page ] end -->
        <!-- [ Main Content ] end -->
    </div>
@endsection
@push('scripts')
    <script>
        $(document).ready(function() {
            var type = '{{ $project->type }}';
            if (type == 'template') {
                $('.pro_type').addClass('d-none');
            } else {
                $('.pro_type').removeClass('d-none');
            }
        });
    </script>
    <script src="{{ asset('assets/js/plugins/apexcharts.min.js') }}"></script>
    <script src="{{ asset('js/letter.avatar.js') }}"></script>
    <script src="{{ asset('packages/workdo/Taskly/src/Resources/assets/js/dropzone.min.js') }}"></script>
    <script>
        Dropzone.autoDiscover = false;
        myDropzone = new Dropzone("#dropzonewidget", {
            maxFiles: 20,
            maxFilesize: 20,
            parallelUploads: 1,
            acceptedFiles: ".jpeg,.jpg,.png,.pdf,.doc,.txt",
            url: "{{ route('projects.file.upload', [$project->id]) }}",
            success: function(file, response) {
                if (response.is_success) {
                    dropzoneBtn(file, response);
                    toastrs('{{ __('Success') }}', 'File Successfully Uploaded', 'success');
                } else {
                    myDropzone.removeFile(response.error);
                    toastrs('Error', response.error, 'error');
                }
            },
            error: function(file, response) {
                myDropzone.removeFile(file);
                if (response.error) {
                    toastrs('Error', response.error, 'error');
                } else {
                    toastrs('Error', response, 'error');
                }
            }
        });
        myDropzone.on("sending", function(file, xhr, formData) {
            formData.append("_token", $('meta[name="csrf-token"]').attr('content'));
            formData.append("project_id", {{ $project->id }});
        });

        @if (isset($permisions) && in_array('show uploading', $permisions))
            $(".dz-hidden-input").prop("disabled", true);
            myDropzone.removeEventListeners();
        @endif

        function dropzoneBtn(file, response) {

            var html = document.createElement('div');
            var download = document.createElement('a');
            download.setAttribute('href', response.download);
            download.setAttribute('class', "action-btn btn-primary mx-1  btn btn-sm d-inline-flex align-items-center");
            download.setAttribute('data-toggle', "tooltip");
            download.setAttribute('download', file.name);
            download.setAttribute('title', "{{ __('Download') }}");
            download.innerHTML = "<i class='ti ti-download'> </i>";
            html.appendChild(download);

            @if (isset($permisions) && in_array('show uploading', $permisions))
            @else
                var del = document.createElement('a');
                del.setAttribute('href', response.delete);
                del.setAttribute('class', "action-btn btn-danger mx-1  btn btn-sm d-inline-flex align-items-center");
                del.setAttribute('data-toggle', "popover");
                del.setAttribute('title', "{{ __('Delete') }}");
                del.innerHTML = "<i class='ti ti-trash '></i>";

                del.addEventListener("click", function(e) {
                    e.preventDefault();
                    e.stopPropagation();

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
                            url: btn.attr('href'),
                            type: 'DELETE',
                            success: function(response) {
                                if (response.is_success) {
                                    btn.closest('.dz-image-preview').remove();
                                    btn.closest('.dz-file-preview').remove();
                                    toastrs('{{ __('Success') }}', 'File Successfully Deleted',
                                        'success');
                                } else {
                                    toastrs('{{ __('Error') }}', 'Something Wents Wrong.', 'error');
                                }
                            },
                            error: function(response) {
                                response = response.responseJSON;
                                if (response.is_success) {
                                    toastrs('{{ __('Error') }}', 'Something Wents Wrong.', 'error');
                                } else {
                                    toastrs('{{ __('Error') }}', 'Something Wents Wrong.', 'error');
                                }
                            }
                        })
                    }
                });

            });
            html.appendChild(del);

            @endif

            file.previewTemplate.appendChild(html);
        }

        @php($files = $project->files)
        @foreach ($files as $file)

            @php($storage_file = get_base_file($file->file_path))
            // Create the mock file:
            var mockFile = {
                name: "{{ $file->file_name }}",
                size: "{{ get_size(get_file($file->file_path)) }}"
            };
            // Call the default addedfile event handler
            myDropzone.emit("addedfile", mockFile);
            // And optionally show the thumbnail of the file:
            myDropzone.emit("thumbnail", mockFile, "{{ get_file($file->file_path) }}");
            myDropzone.emit("complete", mockFile);

            dropzoneBtn(mockFile, {
                download: "{{ get_file($file->file_path) }}",
                delete: "{{ route('projects.file.delete', [$project->id, $file->id]) }}"
            });
        @endforeach
    </script>
    <script>
        (function() {
            var options = {
                chart: {
                    height: 200,
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

        $('.cp_link').on('click', function() {
            var value = $(this).attr('data-link');
            var $temp = $("<input>");
            $("body").append($temp);
            $temp.val(value).select();
            document.execCommand("copy");
            $temp.remove();
            toastrs('success', '{{ __('Link Copy on Clipboard') }}', 'success')
        });
    </script>
@endpush
