@extends('layouts.layoutMaster')

@section('title', __('Timesheets'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-responsive-bs5/responsive.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-buttons-bs5/buttons.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/sweetalert2/sweetalert2.scss'])
    @vite(['resources/assets/vendor/libs/select2/select2.scss'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables-bootstrap5.js'])
    @vite(['resources/assets/vendor/libs/sweetalert2/sweetalert2.js'])
    @vite(['resources/assets/vendor/libs/select2/select2.js'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.js'])
@endsection

@section('page-script')
    <script>
        window.pageData = {
            urls: {
                datatableUrl: @json(route('pmcore.timesheets.data')),
                createUrl: @json(route('pmcore.timesheets.create')),
                editUrl: @json(route('pmcore.timesheets.edit', ['timesheet' => '__ID__'])),
                deleteUrl: @json(route('pmcore.timesheets.destroy', ['timesheet' => '__ID__'])),
                approveUrl: @json(route('pmcore.timesheets.approve', ['timesheet' => '__ID__'])),
                rejectUrl: @json(route('pmcore.timesheets.reject', ['timesheet' => '__ID__'])),
                submitUrl: @json(route('pmcore.timesheets.submit', ['timesheet' => '__ID__'])),
                statisticsUrl: @json(route('pmcore.timesheets.statistics')),
                projectTasksUrl: @json(route('pmcore.timesheets.project-tasks', ['project' => '__PROJECT_ID__'])),
                usersSearchUrl: @json(route('pmcore.users.search')),
                projectsSearchUrl: @json(route('pmcore.projects.search'))
            },
            labels: {
                confirmDelete: @json(__('Are you sure you want to delete this timesheet?')),
                confirmApprove: @json(__('Are you sure you want to approve this timesheet?')),
                confirmReject: @json(__('Are you sure you want to reject this timesheet?')),
                confirmSubmit: @json(__('Are you sure you want to submit this timesheet for approval?')),
                deleteSuccess: @json(__('Timesheet deleted successfully!')),
                approveSuccess: @json(__('Timesheet approved successfully!')),
                rejectSuccess: @json(__('Timesheet rejected successfully!')),
                submitSuccess: @json(__('Timesheet submitted for approval!')),
                error: @json(__('An error occurred. Please try again.')),
                loading: @json(__('Loading...')),
                selectProject: @json(__('Select a project first')),
                noTasksFound: @json(__('No tasks found for this project'))
            }
        };
    </script>
    @vite(['Modules/PMCore/resources/assets/js/timesheets.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Timesheets')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Timesheets'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<!-- Statistics Cards -->
<div class="row mb-4">
    <div class="col-lg-3 col-md-6 col-12">
        <div class="card h-100">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div class="card-info">
                        <p class="card-text">{{ __('Total Hours') }}</p>
                        <div class="d-flex align-items-end mb-2">
                            <h4 class="card-title mb-0 me-2" id="totalHours">0</h4>
                            <small class="text-muted">hrs</small>
                        </div>
                    </div>
                    <div class="card-icon">
                        <span class="badge bg-label-primary rounded p-2">
                            <i class="bx bx-time-five bx-sm"></i>
                        </span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-12">
        <div class="card h-100">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div class="card-info">
                        <p class="card-text">{{ __('Billable Hours') }}</p>
                        <div class="d-flex align-items-end mb-2">
                            <h4 class="card-title mb-0 me-2" id="billableHours">0</h4>
                            <small class="text-muted">hrs</small>
                        </div>
                    </div>
                    <div class="card-icon">
                        <span class="badge bg-label-success rounded p-2">
                            <i class="bx bx-dollar-circle bx-sm"></i>
                        </span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-12">
        <div class="card h-100">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div class="card-info">
                        <p class="card-text">{{ __('Approved Hours') }}</p>
                        <div class="d-flex align-items-end mb-2">
                            <h4 class="card-title mb-0 me-2" id="approvedHours">0</h4>
                            <small class="text-muted">hrs</small>
                        </div>
                    </div>
                    <div class="card-icon">
                        <span class="badge bg-label-success rounded p-2">
                            <i class="bx bx-check-circle bx-sm"></i>
                        </span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-12">
        <div class="card h-100">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div class="card-info">
                        <p class="card-text">{{ __('Pending Hours') }}</p>
                        <div class="d-flex align-items-end mb-2">
                            <h4 class="card-title mb-0 me-2" id="pendingHours">0</h4>
                            <small class="text-muted">hrs</small>
                        </div>
                    </div>
                    <div class="card-icon">
                        <span class="badge bg-label-warning rounded p-2">
                            <i class="bx bx-hourglass bx-sm"></i>
                        </span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Filters -->
<div class="card mb-4">
    <div class="card-header">
        <h5 class="card-title mb-0">{{ __('Filters') }}</h5>
    </div>
    <div class="card-body">
        <form id="filterForm">
            <div class="row">
                <div class="col-md-3">
                    <div class="mb-3">
                        <label for="filterUser" class="form-label">{{ __('User') }}</label>
                        <select class="form-select" id="filterUser" name="user_id">
                            <option value="">{{ __('All Users') }}</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="mb-3">
                        <label for="filterProject" class="form-label">{{ __('Project') }}</label>
                        <select class="form-select" id="filterProject" name="project_id">
                            <option value="">{{ __('All Projects') }}</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="mb-3">
                        <label for="filterStatus" class="form-label">{{ __('Status') }}</label>
                        <select class="form-select" id="filterStatus" name="status">
                            <option value="">{{ __('All Statuses') }}</option>
                            <option value="draft">{{ __('Draft') }}</option>
                            <option value="submitted">{{ __('Submitted') }}</option>
                            <option value="approved">{{ __('Approved') }}</option>
                            <option value="rejected">{{ __('Rejected') }}</option>
                            <option value="invoiced">{{ __('Invoiced') }}</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="mb-3">
                        <label for="filterDateFrom" class="form-label">{{ __('From Date') }}</label>
                        <input type="text" class="form-control" id="filterDateFrom" name="date_from" placeholder="YYYY-MM-DD">
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="mb-3">
                        <label for="filterDateTo" class="form-label">{{ __('To Date') }}</label>
                        <input type="text" class="form-control" id="filterDateTo" name="date_to" placeholder="YYYY-MM-DD">
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-12">
                    <button type="submit" class="btn btn-primary">
                        <i class="bx bx-filter-alt me-1"></i>{{ __('Apply Filters') }}
                    </button>
                    <button type="button" class="btn btn-label-secondary" id="clearFilters">
                        <i class="bx bx-x me-1"></i>{{ __('Clear Filters') }}
                    </button>
                </div>
            </div>
        </form>
    </div>
</div>

<!-- Timesheets Table -->
<div class="card">
    <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="mb-0">{{ __('Timesheets') }}</h5>
            <a href="{{ route('pmcore.timesheets.create') }}" class="btn btn-primary">
                <i class="bx bx-plus me-1"></i>{{ __('Add Timesheet') }}
            </a>
        </div>
        <div class="card-datatable table-responsive">
            <table class="table table-hover border-top datatables-timesheets">
                <thead>
                    <tr>
                        <th>{{ __('User') }}</th>
                        <th>{{ __('Project') }}</th>
                        <th>{{ __('Task') }}</th>
                        <th>{{ __('Date') }}</th>
                        <th>{{ __('Hours') }}</th>
                        <th>{{ __('Description') }}</th>
                        <th>{{ __('Amount') }}</th>
                        <th>{{ __('Status') }}</th>
                        <th>{{ __('Approved By') }}</th>
                        <th>{{ __('Actions') }}</th>
                    </tr>
                </thead>
            </table>
        </div>
    </div>
</div>

@endsection
