@extends('layouts.layoutMaster')

@section('title', __('Project Timesheets') . ' - ' . $project->name)

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-responsive-bs5/responsive.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-buttons-bs5/buttons.bootstrap5.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables-bootstrap5.js'])
@endsection

@section('page-script')
    <script>
        window.pageData = {
            urls: {
                datatable: @json(route('pmcore.timesheets.data')),
                create: @json(route('pmcore.timesheets.create')),
                approve: @json(route('pmcore.timesheets.approve', ':id')),
                reject: @json(route('pmcore.timesheets.reject', ':id')),
                submit: @json(route('pmcore.timesheets.submit', ':id'))
            },
            labels: {
                confirmApprove: @json(__('Are you sure you want to approve this timesheet?')),
                confirmReject: @json(__('Are you sure you want to reject this timesheet?')),
                confirmSubmit: @json(__('Are you sure you want to submit this timesheet for approval?')),
                success: @json(__('Success!')),
                error: @json(__('Error!')),
                approved: @json(__('Timesheet approved successfully!')),
                rejected: @json(__('Timesheet rejected successfully!')),
                submitted: @json(__('Timesheet submitted for approval!'))
            },
            projectId: {{ $project->id }}
        };
    </script>
    @vite(['Modules/PMCore/resources/assets/js/project-timesheets.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Project Timesheets')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Projects'), 'url' => route('pmcore.projects.index')],
        ['name' => $project->name, 'url' => route('pmcore.projects.show', $project->id)],
        ['name' => __('Timesheets'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<!-- Timesheet Statistics -->
<div class="row mb-4">
    <div class="col-lg-2 col-md-4 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-primary rounded">
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
    <div class="col-lg-2 col-md-4 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-success rounded">
                            <i class="bx bx-money text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Total Value') }}</div>
                        <h5 class="card-title mb-0">{{ \App\Helpers\FormattingHelper::formatCurrency($stats['total_amount']) }}</h5>
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
                            <i class="bx bx-check-circle text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Approved') }}</div>
                        <h5 class="card-title mb-0">{{ number_format($stats['approved_hours'], 2) }}</h5>
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
                            <i class="bx bx-hourglass text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Pending') }}</div>
                        <h5 class="card-title mb-0">{{ number_format($stats['pending_hours'], 2) }}</h5>
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
                            <i class="bx bx-group text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Contributors') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['users_count'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Project Info -->
<div class="row">
    <div class="col-12">
        <div class="card mb-4">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h5 class="mb-1">{{ $project->name }}</h5>
                        <p class="text-muted mb-0">{{ $project->code }}</p>
                    </div>
                    <div>
                        <a href="{{ route('pmcore.projects.show', $project->id) }}" class="btn btn-label-primary">
                            <i class="bx bx-arrow-back me-1"></i>{{ __('Back to Project') }}
                        </a>
                        @can('create', \Modules\PMCore\app\Models\Timesheet::class)
                            <a href="{{ route('pmcore.timesheets.create', ['project_id' => $project->id]) }}" class="btn btn-primary">
                                <i class="bx bx-plus me-1"></i>{{ __('New Timesheet') }}
                            </a>
                        @endcan
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Timesheets List -->
<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-body">
                <div class="table-responsive">
                    <table id="timesheetsTable" class="table table-bordered table-hover">
                        <thead>
                            <tr>
                                <th>{{ __('Date') }}</th>
                                <th>{{ __('User') }}</th>
                                <th>{{ __('Task') }}</th>
                                <th>{{ __('Description') }}</th>
                                <th>{{ __('Hours') }}</th>
                                <th>{{ __('Billable') }}</th>
                                <th>{{ __('Status') }}</th>
                                <th>{{ __('Amount') }}</th>
                                <th>{{ __('Actions') }}</th>
                            </tr>
                        </thead>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>

@endsection
