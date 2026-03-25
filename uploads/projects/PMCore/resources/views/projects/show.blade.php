@extends('layouts.layoutMaster')

@section('title', __('Project Details'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/apex-charts/apex-charts.scss'])
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-responsive-bs5/responsive.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/select2/select2.scss'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.scss'])
    @vite(['resources/assets/vendor/libs/sweetalert2/sweetalert2.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/apex-charts/apexcharts.js'])
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables-bootstrap5.js'])
    @vite(['resources/assets/vendor/libs/select2/select2.js'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.js'])
    @vite(['resources/assets/vendor/libs/sweetalert2/sweetalert2.js'])
@endsection

@section('page-script')
    @vite(['Modules/PMCore/resources/assets/js/project-show.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="$project->name"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Projects'), 'url' => route('pmcore.projects.index')],
        ['name' => $project->name, 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<div class="row">
    <div class="col-md-12">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <div class="d-flex gap-2">
                <a href="{{ route('pmcore.projects.tasks.index', $project->id) }}" class="btn btn-outline-primary">
                    <i class="bx bx-task me-1"></i>{{ __('View Tasks') }}
                </a>
                <a href="{{ route('pmcore.projects.timesheets', $project->id) }}" class="btn btn-outline-primary">
                    <i class="bx bx-time-five me-1"></i>{{ __('View Timesheets') }}
                </a>
                <a href="{{ route('pmcore.resources.index', ['project_id' => $project->id]) }}" class="btn btn-outline-primary">
                    <i class="bx bx-user-pin me-1"></i>{{ __('View Resources') }}
                </a>
                <a href="{{ route('pmcore.projects.edit', $project->id) }}" class="btn btn-primary">
                    <i class="bx bx-edit me-1"></i>{{ __('Edit Project') }}
                </a>
                <div class="btn-group">
                    <button type="button" class="btn btn-outline-secondary dropdown-toggle" data-bs-toggle="dropdown">
                        <i class="bx bx-dots-vertical-rounded"></i>
                    </button>
                    <ul class="dropdown-menu">
                        <li><a class="dropdown-item duplicate-project" href="#" data-id="{{ $project->id }}"><i class="bx bx-copy me-1"></i>{{ __('Duplicate') }}</a></li>
                        <li><a class="dropdown-item archive-project" href="#" data-id="{{ $project->id }}"><i class="bx bx-archive me-1"></i>{{ __('Archive') }}</a></li>
                        <li><hr class="dropdown-divider"></li>
                        <li><a class="dropdown-item text-danger delete-project" href="#" data-id="{{ $project->id }}"><i class="bx bx-trash me-1"></i>{{ __('Delete') }}</a></li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Project Overview Cards -->
<div class="row mb-4">
    <div class="col-lg-2 col-md-4 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-primary rounded">
                            <i class="bx bx-group text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Team Members') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['total_members'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-md-4 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-info rounded">
                            <i class="bx bx-task text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Total Tasks') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['total_tasks'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-md-4 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-success rounded">
                            <i class="bx bx-check-circle text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Completed') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['completed_tasks'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-md-4 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-warning rounded">
                            <i class="bx bx-time-five text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('In Progress') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['in_progress_tasks'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-md-4 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-danger rounded">
                            <i class="bx bx-alarm-exclamation text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Overdue') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['overdue_tasks'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-md-4 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-info rounded">
                            <i class="bx bx-time-five text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Total Hours') }}</div>
                        <h5 class="card-title mb-0">{{ number_format($stats['total_hours'], 2) }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-md-4 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-dark rounded">
                            <i class="bx bx-flag text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Milestones') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['milestones'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Task Progress and Summary Row -->
<div class="row mb-4">
    <div class="col-md-12">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">{{ __('Project Progress') }}</h5>
                <div class="d-flex align-items-center">
                    <div class="small text-muted me-3">
                        @if($stats['days_remaining'] !== null)
                            {{ $stats['days_remaining'] >= 0 ? $stats['days_remaining'] . ' ' . __('days remaining') : __('Overdue') }}
                        @else
                            {{ __('No end date set') }}
                        @endif
                    </div>
                    <span class="badge bg-label-{{ $stats['is_overdue'] ? 'danger' : 'primary' }}">
                        {{ $stats['progress_percentage'] }}% {{ __('Complete') }}
                    </span>
                </div>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-8">
                        <div class="progress mb-3" style="height: 10px;">
                            <div class="progress-bar" role="progressbar" style="width: {{ $stats['progress_percentage'] }}%"></div>
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <h6 class="mb-2">{{ __('Task Status Distribution') }}</h6>
                                @if(!empty($stats['task_stats_by_status']))
                                    @foreach($stats['task_stats_by_status'] as $status)
                                    <div class="d-flex justify-content-between align-items-center mb-2">
                                        <div class="d-flex align-items-center">
                                            <span class="badge badge-dot me-2" style="background-color: {{ $status['color'] }}"></span>
                                            <span class="small">{{ $status['name'] }}</span>
                                        </div>
                                        <span class="small text-muted">{{ $status['count'] }}</span>
                                    </div>
                                    @endforeach
                                @else
                                    <p class="text-muted small">{{ __('No tasks found') }}</p>
                                @endif
                            </div>
                            <div class="col-md-6">
                                <h6 class="mb-2">{{ __('Priority Distribution') }}</h6>
                                @if(!empty($stats['task_stats_by_priority']))
                                    @foreach($stats['task_stats_by_priority'] as $priority)
                                    <div class="d-flex justify-content-between align-items-center mb-2">
                                        <div class="d-flex align-items-center">
                                            <span class="badge badge-dot me-2" style="background-color: {{ $priority['color'] }}"></span>
                                            <span class="small">{{ $priority['name'] }}</span>
                                        </div>
                                        <span class="small text-muted">{{ $priority['count'] }}</span>
                                    </div>
                                    @endforeach
                                @else
                                    <p class="text-muted small">{{ __('No tasks found') }}</p>
                                @endif
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="border-start ps-3">
                            <h6 class="mb-3">{{ __('Quick Stats') }}</h6>
                            <div class="row g-3">
                                <div class="col-6">
                                    <div class="text-center">
                                        <h4 class="mb-1 text-primary">{{ $stats['total_tasks'] }}</h4>
                                        <p class="small mb-0 text-muted">{{ __('Total') }}</p>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="text-center">
                                        <h4 class="mb-1 text-success">{{ $stats['completed_tasks'] }}</h4>
                                        <p class="small mb-0 text-muted">{{ __('Completed') }}</p>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="text-center">
                                        <h4 class="mb-1 text-warning">{{ $stats['pending_tasks'] }}</h4>
                                        <p class="small mb-0 text-muted">{{ __('Pending') }}</p>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="text-center">
                                        <h4 class="mb-1 text-danger">{{ $stats['overdue_tasks'] }}</h4>
                                        <p class="small mb-0 text-muted">{{ __('Overdue') }}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Timesheet Stats -->
<div class="row mb-4">
    <div class="col-lg-3 col-md-6 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-info rounded">
                            <i class="bx bx-dollar-circle text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Billable Hours') }}</div>
                        <h5 class="card-title mb-0">{{ number_format($stats['billable_hours'], 2) }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-success rounded">
                            <i class="bx bx-money text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Timesheet Value') }}</div>
                        <h5 class="card-title mb-0">{{ \App\Helpers\FormattingHelper::formatCurrency($stats['total_timesheet_amount']) }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-warning rounded">
                            <i class="bx bx-check-circle text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Approved Hours') }}</div>
                        <h5 class="card-title mb-0">{{ number_format($stats['approved_hours'], 2) }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-danger rounded">
                            <i class="bx bx-dollar text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Avg Rate/Hour') }}</div>
                        <h5 class="card-title mb-0">
                            @if($stats['billable_hours'] > 0)
                                {{ \App\Helpers\FormattingHelper::formatCurrency($stats['total_timesheet_amount'] / $stats['billable_hours']) }}
                            @else
                                -
                            @endif
                        </h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Project Details -->
<div class="row">
    <div class="col-md-8">
        <!-- Project Information -->
        <div class="card mb-4">
            <div class="card-header">
                <h5 class="card-title mb-0">{{ __('Project Information') }}</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <div class="mb-3">
                            <label class="form-label">{{ __('Project Name') }}</label>
                            <p class="mb-0">{{ $project->name }}</p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">{{ __('Code') }}</label>
                            <p class="mb-0">{{ $project->code ?: '-' }}</p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">{{ __('Client') }}</label>
                            <p class="mb-0">{{ $project->client?->name ?? '-' }}</p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">{{ __('Project Manager') }}</label>
                            <p class="mb-0">{{ $project->projectManager?->name ?? '-' }}</p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="mb-3">
                            <label class="form-label">{{ __('Status') }}</label>
                            <p class="mb-0">
                                <span class="badge bg-label-{{ $project->status->color() }}">
                                    {{ $project->status->label() }}
                                </span>
                            </p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">{{ __('Priority') }}</label>
                            <p class="mb-0">
                                <span class="badge bg-label-{{ $project->priority->color() }}">
                                    {{ $project->priority->label() }}
                                </span>
                            </p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">{{ __('Start Date') }}</label>
                            <p class="mb-0">{{ $project->start_date?->format('M d, Y') ?? '-' }}</p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">{{ __('End Date') }}</label>
                            <p class="mb-0">{{ $project->end_date?->format('M d, Y') ?? '-' }}</p>
                        </div>
                    </div>
                </div>
                @if($project->description)
                <div class="mb-3">
                    <label class="form-label">{{ __('Description') }}</label>
                    <p class="mb-0">{{ $project->description }}</p>
                </div>
                @endif
            </div>
        </div>

        <!-- Progress Section -->
        <div class="card mb-4">
            <div class="card-header">
                <h5 class="card-title mb-0">{{ __('Project Progress') }}</h5>
            </div>
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span>{{ __('Overall Progress') }}</span>
                    <span class="fw-bold">{{ $stats['progress_percentage'] }}%</span>
                </div>
                <div class="progress mb-3" style="height: 10px;">
                    <div class="progress-bar" role="progressbar" style="width: {{ $stats['progress_percentage'] }}%"></div>
                </div>
                <div class="row text-center">
                    <div class="col-4">
                        <div class="border rounded p-2">
                            <div class="fw-bold">{{ $stats['total_tasks'] }}</div>
                            <small class="text-muted">{{ __('Total Tasks') }}</small>
                        </div>
                    </div>
                    <div class="col-4">
                        <div class="border rounded p-2">
                            <div class="fw-bold">{{ $stats['completed_tasks'] }}</div>
                            <small class="text-muted">{{ __('Completed') }}</small>
                        </div>
                    </div>
                    <div class="col-4">
                        <div class="border rounded p-2">
                            <div class="fw-bold">{{ $stats['total_tasks'] - $stats['completed_tasks'] }}</div>
                            <small class="text-muted">{{ __('Remaining') }}</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="col-md-4">
        <!-- Budget Information -->
        <div class="card mb-4">
            <div class="card-header">
                <h5 class="card-title mb-0">{{ __('Budget Information') }}</h5>
            </div>
            <div class="card-body">
                <div class="mb-3">
                    <div class="d-flex justify-content-between">
                        <span>{{ __('Total Budget') }}</span>
                        <span class="fw-bold">${{ number_format($project->budget ?? 0, 2) }}</span>
                    </div>
                </div>
                <div class="mb-3">
                    <div class="d-flex justify-content-between">
                        <span>{{ __('Hourly Rate') }}</span>
                        <span class="fw-bold">${{ number_format($project->hourly_rate ?? 0, 2) }}</span>
                    </div>
                </div>
                <div class="mb-3">
                    <div class="d-flex justify-content-between">
                        <span>{{ __('Billable') }}</span>
                        <span class="badge bg-label-{{ $project->is_billable ? 'success' : 'secondary' }}">
                            {{ $project->is_billable ? __('Yes') : __('No') }}
                        </span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Team Members -->
        <div class="card mb-4">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">{{ __('Team Members') }}</h5>
                <button class="btn btn-sm btn-outline-primary" data-bs-toggle="modal" data-bs-target="#addMemberModal">
                    <i class="bx bx-plus me-1"></i>{{ __('Add Member') }}
                </button>
            </div>
            <div class="card-body">
                @if($project->activeMembers->count() > 0)
                    @foreach($project->activeMembers as $member)
                        <div class="d-flex align-items-center mb-3">
                            <div class="avatar avatar-sm me-3">
                                <div class="avatar-initial bg-label-primary rounded-circle">
                                    {{ substr($member->user->getFullName() ?? 'N/A', 0, 2) }}
                                </div>
                            </div>
                            <div class="flex-grow-1">
                                <h6 class="mb-0">{{ $member->user->getFullName() ?? 'Unknown User' }}</h6>
                                <small class="text-muted">{{ $member->role?->label() ?? $member->role }}</small>
                            </div>
                            <div class="dropdown">
                                <button class="btn btn-sm btn-icon" data-bs-toggle="dropdown">
                                    <i class="bx bx-dots-vertical-rounded"></i>
                                </button>
                                <ul class="dropdown-menu dropdown-menu-end">
                                    @can('manageTeam', $project)
                                    <li><a class="dropdown-item edit-member-role" href="#" data-member-id="{{ $member->id }}">{{ __('Edit Role') }}</a></li>
                                    <li><a class="dropdown-item text-danger remove-member" href="#" data-member-id="{{ $member->id }}">{{ __('Remove') }}</a></li>
                                    @else
                                    <li><span class="dropdown-item-text text-muted">{{ __('No actions available') }}</span></li>
                                    @endcan
                                </ul>
                            </div>
                        </div>
                    @endforeach
                @else
                    <div class="text-center py-3">
                        <i class="bx bx-group bx-lg text-muted"></i>
                        <p class="text-muted mt-2">{{ __('No team members assigned') }}</p>
                    </div>
                @endif
            </div>
        </div>

        <!-- Resource Allocations -->
        <div class="card mb-4">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">{{ __('Resource Allocations') }}</h5>
                <a href="{{ route('pmcore.resources.create', ['project_id' => $project->id]) }}" class="btn btn-sm btn-outline-primary">
                    <i class="bx bx-plus me-1"></i>{{ __('Allocate Resource') }}
                </a>
            </div>
            <div class="card-body">
                @if($project->resourceAllocations->count() > 0)
                    @foreach($project->resourceAllocations as $allocation)
                        <div class="d-flex align-items-center mb-3">
                            <div class="avatar avatar-sm me-3">
                                @if($allocation->user->profile_picture)
                                    <img src="{{ $allocation->user->getProfilePicture() }}" alt="Avatar" class="rounded-circle" />
                                @else
                                    <div class="avatar-initial bg-label-primary rounded-circle">
                                        {{ $allocation->user->getInitials() }}
                                    </div>
                                @endif
                            </div>
                            <div class="flex-grow-1">
                                <h6 class="mb-0">{{ $allocation->user->name }}</h6>
                                <div class="d-flex align-items-center gap-2">
                                    <small class="text-muted">
                                        {{ $allocation->allocation_percentage }}% allocation
                                    </small>
                                    <span class="badge bg-label-{{ $allocation->status === 'active' ? 'success' : 'warning' }}">
                                        {{ ucfirst($allocation->status) }}
                                    </span>
                                    @if($allocation->is_billable)
                                        <span class="badge bg-label-info">{{ __('Billable') }}</span>
                                    @endif
                                </div>
                                <small class="text-muted">
                                    {{ $allocation->start_date->format('M d, Y') }}
                                    @if($allocation->end_date)
                                        - {{ $allocation->end_date->format('M d, Y') }}
                                    @else
                                        - {{ __('Ongoing') }}
                                    @endif
                                </small>
                            </div>
                            <div class="dropdown">
                                <button class="btn btn-sm btn-icon" data-bs-toggle="dropdown">
                                    <i class="bx bx-dots-vertical-rounded"></i>
                                </button>
                                <ul class="dropdown-menu dropdown-menu-end">
                                    <li>
                                        <a class="dropdown-item" href="{{ route('pmcore.resources.schedule', $allocation->user_id) }}">
                                            <i class="bx bx-calendar me-1"></i>{{ __('View Schedule') }}
                                        </a>
                                    </li>
                                    <li>
                                        <a class="dropdown-item" href="{{ route('pmcore.resources.edit', $allocation->id) }}">
                                            <i class="bx bx-edit me-1"></i>{{ __('Edit Allocation') }}
                                        </a>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    @endforeach

                    <!-- Resource Summary -->
                    <div class="border-top pt-3 mt-3">
                        <div class="row text-center">
                            <div class="col-4">
                                <h6 class="mb-1">{{ $stats['total_allocated_resources'] }}</h6>
                                <small class="text-muted">{{ __('Resources') }}</small>
                            </div>
                            <div class="col-4">
                                <h6 class="mb-1">{{ $stats['resource_hours_per_day'] }}h</h6>
                                <small class="text-muted">{{ __('Hours/Day') }}</small>
                            </div>
                            <div class="col-4">
                                <h6 class="mb-1">{{ number_format($stats['total_allocation_percentage'] / max($stats['total_allocated_resources'], 1), 0) }}%</h6>
                                <small class="text-muted">{{ __('Avg Allocation') }}</small>
                            </div>
                        </div>
                    </div>
                @else
                    <div class="text-center py-3">
                        <i class="bx bx-user-pin bx-lg text-muted"></i>
                        <p class="text-muted mt-2">{{ __('No resources allocated') }}</p>
                        <a href="{{ route('pmcore.resources.create', ['project_id' => $project->id]) }}" class="btn btn-sm btn-primary mt-2">
                            {{ __('Allocate First Resource') }}
                        </a>
                    </div>
                @endif
            </div>
        </div>
    </div>
</div>

<!-- Task Summary Section -->
<div class="row mb-4">
    <div class="col-md-6">
        <!-- Recent Tasks -->
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">{{ __('Recent Tasks') }}</h5>
                <a href="{{ route('pmcore.projects.tasks.index', $project->id) }}" class="btn btn-sm btn-outline-primary">
                    {{ __('View All') }}
                </a>
            </div>
            <div class="card-body">
                @if(!empty($stats['recent_tasks']))
                    @foreach($stats['recent_tasks'] as $task)
                    <div class="d-flex align-items-center mb-3">
                        <div class="flex-shrink-0">
                            @if($task['is_milestone'])
                                <i class="bx bx-flag text-warning"></i>
                            @else
                                <i class="bx bx-task text-muted"></i>
                            @endif
                        </div>
                        <div class="flex-grow-1 ms-3">
                            <div class="d-flex justify-content-between align-items-start">
                                <div>
                                    <h6 class="mb-1">{{ $task['title'] }}</h6>
                                    <div class="d-flex align-items-center gap-2">
                                        @if($task['status'])
                                        <span class="badge badge-sm" style="background-color: {{ $task['status_color'] }}; color: white;">
                                            {{ $task['status'] }}
                                        </span>
                                        @endif
                                        @if($task['priority'])
                                        <span class="badge badge-sm" style="background-color: {{ $task['priority_color'] }}; color: white;">
                                            {{ $task['priority'] }}
                                        </span>
                                        @endif
                                        @if($task['is_overdue'])
                                        <span class="badge bg-danger">{{ __('Overdue') }}</span>
                                        @endif
                                    </div>
                                    @if($task['assigned_to'])
                                    <small class="text-muted">{{ __('Assigned to') }}: {{ $task['assigned_to'] }}</small>
                                    @endif
                                </div>
                                <div class="text-end">
                                    @if($task['completed_at'])
                                    <small class="text-success">{{ __('Completed') }}: {{ $task['completed_at'] }}</small>
                                    @elseif($task['due_date'])
                                    <small class="text-muted">{{ __('Due') }}: {{ $task['due_date'] }}</small>
                                    @endif
                                </div>
                            </div>
                        </div>
                    </div>
                    @endforeach
                @else
                    <div class="text-center py-3">
                        <i class="bx bx-task bx-lg text-muted"></i>
                        <p class="text-muted mt-2">{{ __('No recent tasks') }}</p>
                    </div>
                @endif
            </div>
        </div>
    </div>

    <div class="col-md-6">
        <!-- Upcoming Tasks -->
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">{{ __('Upcoming Tasks') }}</h5>
                <a href="{{ route('pmcore.projects.tasks.board', $project->id) }}" class="btn btn-sm btn-outline-primary">
                    {{ __('Board View') }}
                </a>
            </div>
            <div class="card-body">
                @if(!empty($stats['upcoming_tasks']))
                    @foreach($stats['upcoming_tasks'] as $task)
                    <div class="d-flex align-items-center mb-3">
                        <div class="flex-shrink-0">
                            @if($task['is_milestone'])
                                <i class="bx bx-flag text-warning"></i>
                            @else
                                <i class="bx bx-task text-muted"></i>
                            @endif
                        </div>
                        <div class="flex-grow-1 ms-3">
                            <div class="d-flex justify-content-between align-items-start">
                                <div>
                                    <h6 class="mb-1">{{ $task['title'] }}</h6>
                                    <div class="d-flex align-items-center gap-2">
                                        @if($task['status'])
                                        <span class="badge badge-sm" style="background-color: {{ $task['status_color'] }}; color: white;">
                                            {{ $task['status'] }}
                                        </span>
                                        @endif
                                        @if($task['priority'])
                                        <span class="badge badge-sm" style="background-color: {{ $task['priority_color'] }}; color: white;">
                                            {{ $task['priority'] }}
                                        </span>
                                        @endif
                                    </div>
                                    @if($task['assigned_to'])
                                    <small class="text-muted">{{ __('Assigned to') }}: {{ $task['assigned_to'] }}</small>
                                    @endif
                                </div>
                                <div class="text-end">
                                    @if($task['due_date'])
                                    <small class="text-muted">{{ __('Due') }}: {{ $task['due_date'] }}</small>
                                    @if($task['days_until_due'] !== null)
                                    <br><small class="text-{{ $task['days_until_due'] <= 3 ? 'warning' : 'muted' }}">
                                        {{ $task['days_until_due'] == 0 ? __('Due Today') : ($task['days_until_due'] == 1 ? __('Due Tomorrow') : $task['days_until_due'] . ' ' . __('days')) }}
                                    </small>
                                    @endif
                                    @endif
                                </div>
                            </div>
                        </div>
                    </div>
                    @endforeach
                @else
                    <div class="text-center py-3">
                        <i class="bx bx-calendar-check bx-lg text-muted"></i>
                        <p class="text-muted mt-2">{{ __('No upcoming tasks') }}</p>
                    </div>
                @endif
            </div>
        </div>
    </div>
</div>

<!-- Recent Timesheets -->
<div class="row">
    <div class="col-12">
        <div class="card mb-4">
            <div class="card-header">
                <div class="d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">{{ __('Recent Timesheets') }}</h5>
                    <a href="{{ route('pmcore.timesheets.index', ['project_id' => $project->id]) }}" class="btn btn-sm btn-label-primary">
                        <i class="bx bx-list-ul me-1"></i>{{ __('View All') }}
                    </a>
                </div>
            </div>
            <div class="card-body">
                @if($recent_timesheets->count() > 0)
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>{{ __('Date') }}</th>
                                    <th>{{ __('User') }}</th>
                                    <th>{{ __('Task') }}</th>
                                    <th>{{ __('Hours') }}</th>
                                    <th>{{ __('Status') }}</th>
                                    <th>{{ __('Amount') }}</th>
                                </tr>
                            </thead>
                            <tbody>
                                @foreach($recent_timesheets as $timesheet)
                                    <tr>
                                        <td>{{ $timesheet->date->format('M d, Y') }}</td>
                                        <td>
                                            <x-datatable-user :user="$timesheet->user" />
                                        </td>
                                        <td>{{ $timesheet->task?->title ?? '-' }}</td>
                                        <td>{{ number_format($timesheet->hours, 2) }}</td>
                                        <td>{!! $timesheet->status_badge !!}</td>
                                        <td>
                                            @if($timesheet->is_billable && $timesheet->billing_rate)
                                                {{ \App\Helpers\FormattingHelper::formatCurrency($timesheet->hours * $timesheet->billing_rate) }}
                                            @else
                                                <span class="text-muted">-</span>
                                            @endif
                                        </td>
                                    </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                @else
                    <div class="text-center py-3">
                        <i class="bx bx-time bx-lg text-muted"></i>
                        <p class="text-muted mt-2">{{ __('No timesheets recorded yet') }}</p>
                    </div>
                @endif
            </div>
        </div>
    </div>
</div>

<!-- Add Member Modal -->
<div class="modal fade" id="addMemberModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">{{ __('Add Team Member') }}</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <form id="addMemberForm">
                <div class="modal-body">
                    <div class="row">
                        <div class="col-12">
                            <div class="mb-3">
                                <label for="member_user_id" class="form-label">{{ __('User') }} <span class="text-danger">*</span></label>
                                <select class="form-select select2" id="member_user_id" name="user_id" required>
                                    <option value="">{{ __('Select User') }}</option>
                                </select>
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="mb-3">
                                <label for="member_role" class="form-label">{{ __('Role') }} <span class="text-danger">*</span></label>
                                <select class="form-select" id="member_role" name="role" required>
                                    @foreach(\Modules\PMCore\app\Enums\ProjectMemberRole::cases() as $role)
                                        @if($role->value !== 'manager')
                                            <option value="{{ $role->value }}">{{ $role->label() }}</option>
                                        @endif
                                    @endforeach
                                </select>
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="mb-3">
                                <label for="member_hourly_rate" class="form-label">{{ __('Hourly Rate') }}</label>
                                <input type="number" class="form-control" id="member_hourly_rate" name="hourly_rate" step="0.01" min="0">
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="mb-3">
                                <label for="member_allocation" class="form-label">{{ __('Allocation %') }}</label>
                                <input type="number" class="form-control" id="member_allocation" name="allocation_percentage" min="0" max="100" value="100">
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">{{ __('Cancel') }}</button>
                    <button type="submit" class="btn btn-primary">{{ __('Add Member') }}</button>
                </div>
            </form>
        </div>
    </div>
</div>

<!-- Edit Project Modal -->
<div class="modal fade" id="editProjectModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-lg modal-dialog-centered">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">{{ __('Edit Project') }}</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <form id="editProjectForm">
                <input type="hidden" id="edit_project_id" name="id">
                <div class="modal-body">
                    <div class="row">
                        <div class="col-12">
                            <div class="mb-3">
                                <label for="edit_project_name" class="form-label">{{ __('Project Name') }} <span class="text-danger">*</span></label>
                                <input type="text" class="form-control" id="edit_project_name" name="name" required>
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <label for="edit_project_code" class="form-label">{{ __('Project Code') }}</label>
                                <input type="text" class="form-control" id="edit_project_code" name="code">
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <label for="edit_project_status" class="form-label">{{ __('Status') }} <span class="text-danger">*</span></label>
                                <select class="form-select" id="edit_project_status" name="status" required>
                                    @foreach(\Modules\PMCore\app\Enums\ProjectStatus::cases() as $status)
                                        <option value="{{ $status->value }}">{{ $status->label() }}</option>
                                    @endforeach
                                </select>
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <label for="edit_project_type" class="form-label">{{ __('Type') }} <span class="text-danger">*</span></label>
                                <select class="form-select" id="edit_project_type" name="type" required>
                                    @foreach(\Modules\PMCore\app\Enums\ProjectType::cases() as $type)
                                        <option value="{{ $type->value }}">{{ $type->label() }}</option>
                                    @endforeach
                                </select>
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <label for="edit_project_priority" class="form-label">{{ __('Priority') }} <span class="text-danger">*</span></label>
                                <select class="form-select" id="edit_project_priority" name="priority" required>
                                    @foreach(\Modules\PMCore\app\Enums\ProjectPriority::cases() as $priority)
                                        <option value="{{ $priority->value }}">{{ $priority->label() }}</option>
                                    @endforeach
                                </select>
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="mb-3">
                                <label for="edit_project_description" class="form-label">{{ __('Description') }}</label>
                                <textarea class="form-control" id="edit_project_description" name="description" rows="3"></textarea>
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <label for="edit_project_client" class="form-label">{{ __('Client') }}</label>
                                <select class="form-select select2" id="edit_project_client" name="client_id">
                                    <option value="">{{ __('Select Client') }}</option>
                                    @if(isset($filters['clients']))
                                        @foreach($filters['clients'] as $client)
                                            <option value="{{ $client->id }}">{{ $client->name }}</option>
                                        @endforeach
                                    @endif
                                </select>
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <label for="edit_project_manager" class="form-label">{{ __('Project Manager') }}</label>
                                <select class="form-select select2" id="edit_project_manager" name="project_manager_id">
                                    <option value="">{{ __('Select Project Manager') }}</option>
                                    @if(isset($filters['project_managers']))
                                        @foreach($filters['project_managers'] as $manager)
                                            <option value="{{ $manager->id }}">{{ $manager->getFullName() }}</option>
                                        @endforeach
                                    @endif
                                </select>
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <label for="edit_project_start_date" class="form-label">{{ __('Start Date') }}</label>
                                <input type="text" class="form-control flatpickr-date" id="edit_project_start_date" name="start_date" placeholder="YYYY-MM-DD">
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <label for="edit_project_end_date" class="form-label">{{ __('End Date') }}</label>
                                <input type="text" class="form-control flatpickr-date" id="edit_project_end_date" name="end_date" placeholder="YYYY-MM-DD">
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <label for="edit_project_budget" class="form-label">{{ __('Budget') }}</label>
                                <input type="number" class="form-control" id="edit_project_budget" name="budget" step="0.01" min="0">
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <label for="edit_project_hourly_rate" class="form-label">{{ __('Hourly Rate') }}</label>
                                <input type="number" class="form-control" id="edit_project_hourly_rate" name="hourly_rate" step="0.01" min="0">
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <label for="edit_project_color" class="form-label">{{ __('Color') }} <span class="text-danger">*</span></label>
                                <input type="color" class="form-control" id="edit_project_color" name="color_code" value="#007bff" required>
                            </div>
                        </div>
                        <div class="col-12 col-md-6">
                            <div class="mb-3">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="edit_project_billable" name="is_billable" value="1">
                                    <label class="form-check-label" for="edit_project_billable">
                                        {{ __('Is Billable') }}
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">{{ __('Cancel') }}</button>
                    <button type="submit" class="btn btn-primary">{{ __('Update Project') }}</button>
                </div>
            </form>
        </div>
    </div>
</div>

<!-- Edit Member Role Modal -->
<div class="modal fade" id="editMemberRoleModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">{{ __('Edit Member Role') }}</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <form id="editMemberRoleForm">
                <input type="hidden" id="edit_member_id" name="member_id">
                <div class="modal-body">
                    <div class="row">
                        <div class="col-12">
                            <div class="mb-3">
                                <label for="edit_member_user_name" class="form-label">{{ __('User') }}</label>
                                <input type="text" class="form-control" id="edit_member_user_name" readonly>
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="mb-3">
                                <label for="edit_member_role_select" class="form-label">{{ __('Role') }} <span class="text-danger">*</span></label>
                                <select class="form-select" id="edit_member_role_select" name="role" required>
                                    @foreach(\Modules\PMCore\app\Enums\ProjectMemberRole::cases() as $role)
                                        @if($role->value !== 'manager')
                                            <option value="{{ $role->value }}">{{ $role->label() }}</option>
                                        @endif
                                    @endforeach
                                </select>
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="mb-3">
                                <label for="edit_member_hourly_rate_input" class="form-label">{{ __('Hourly Rate') }}</label>
                                <input type="number" class="form-control" id="edit_member_hourly_rate_input" name="hourly_rate" step="0.01" min="0">
                            </div>
                        </div>
                        <div class="col-12">
                            <div class="mb-3">
                                <label for="edit_member_allocation_input" class="form-label">{{ __('Allocation %') }}</label>
                                <input type="number" class="form-control" id="edit_member_allocation_input" name="allocation_percentage" min="0" max="100">
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">{{ __('Cancel') }}</button>
                    <button type="submit" class="btn btn-primary">{{ __('Update Role') }}</button>
                </div>
            </form>
        </div>
    </div>
</div>

<script>
const projectData = {
    project: @json($project),
    stats: @json($stats)
};

const pageData = {
    project: @json($project),
    stats: @json($stats),
    urls: {
        projectUpdate: @json(route('pmcore.projects.update', ['project' => $project->id])),
        projectDelete: @json(route('pmcore.projects.destroy', ['project' => $project->id])),
        projectDuplicate: @json(route('pmcore.projects.duplicate', ['project' => $project->id])),
        projectArchive: @json(route('pmcore.projects.archive', ['project' => $project->id])),
        addMember: @json(route('pmcore.projects.members.store', ['project' => $project->id])),
        getMemberDetails: @json(route('pmcore.projects.members.show', ['project' => $project->id, 'member' => '__MEMBER_ID__'])),
        removeMember: @json(route('pmcore.projects.members.destroy', ['project' => $project->id, 'member' => '__MEMBER_ID__'])),
        updateMemberRole: @json(route('pmcore.projects.members.update', ['project' => $project->id, 'member' => '__MEMBER_ID__'])),
        userSearch: @json(route('pmcore.users.search')),
        clientSearch: @json(route('pmcore.clients.search'))
    },
    labels: {
        deleteConfirm: @json(__('Are you sure you want to delete this project?')),
        duplicateConfirm: @json(__('Are you sure you want to duplicate this project?')),
        archiveConfirm: @json(__('Are you sure you want to archive this project?')),
        removeMemberConfirm: @json(__('Are you sure you want to remove this member?')),
        loading: @json(__('Loading...')),
        success: @json(__('Success')),
        error: @json(__('Error')),
        updateSuccess: @json(__('Project updated successfully!')),
        deleteSuccess: @json(__('Project deleted successfully!')),
        duplicateSuccess: @json(__('Project duplicated successfully!')),
        archiveSuccess: @json(__('Project archived successfully!')),
        memberAddSuccess: @json(__('Member added successfully!')),
        memberRemoveSuccess: @json(__('Member removed successfully!')),
        memberRoleUpdateSuccess: @json(__('Member role updated successfully!'))
    }
};
</script>
@endsection
