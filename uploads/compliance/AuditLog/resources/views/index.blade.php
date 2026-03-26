@extends('layouts.layoutMaster')

@section('title', __('Audit Logs'))

@section('vendor-style')
    @vite([
        'resources/assets/vendor/libs/datatables-bs5/datatables.bootstrap5.scss',
        'resources/assets/vendor/libs/datatables-responsive-bs5/responsive.bootstrap5.scss',
        'resources/assets/vendor/libs/flatpickr/flatpickr.scss',
        'resources/assets/vendor/libs/select2/select2.scss',
        'resources/assets/vendor/libs/apex-charts/apex-charts.scss'
    ])
@endsection

@section('vendor-script')
    @vite([
        'resources/assets/vendor/libs/datatables-bs5/datatables-bootstrap5.js',
        'resources/assets/vendor/libs/flatpickr/flatpickr.js',
        'resources/assets/vendor/libs/select2/select2.js',
        'resources/assets/vendor/libs/apex-charts/apexcharts.js'
    ])
@endsection

@section('content')
    <x-breadcrumb
        :title="__('Audit Logs')"
        :items="[]"
        :homeUrl="route('dashboard')"
    />

    <!-- Statistics Cards -->
    <div class="row mb-6">
        <div class="col-lg-3 col-sm-6 mb-6">
            <div class="card h-100">
                <div class="card-body">
                    <div class="d-flex align-items-center mb-2">
                        <div class="avatar me-4">
                            <span class="avatar-initial rounded bg-label-primary">
                                <i class="bx bx-history bx-lg"></i>
                            </span>
                        </div>
                        <h4 class="mb-0" id="totalAudits">0</h4>
                    </div>
                    <p class="mb-0">{{ __('Total Audit Logs') }}</p>
                    <small class="text-muted">{{ __('Last 30 days') }}</small>
                </div>
            </div>
        </div>
        <div class="col-lg-3 col-sm-6 mb-6">
            <div class="card h-100">
                <div class="card-body">
                    <div class="d-flex align-items-center mb-2">
                        <div class="avatar me-4">
                            <span class="avatar-initial rounded bg-label-success">
                                <i class="bx bx-plus-circle bx-lg"></i>
                            </span>
                        </div>
                        <h4 class="mb-0" id="createdCount">0</h4>
                    </div>
                    <p class="mb-0">{{ __('Created Records') }}</p>
                    <small class="text-muted">{{ __('Last 30 days') }}</small>
                </div>
            </div>
        </div>
        <div class="col-lg-3 col-sm-6 mb-6">
            <div class="card h-100">
                <div class="card-body">
                    <div class="d-flex align-items-center mb-2">
                        <div class="avatar me-4">
                            <span class="avatar-initial rounded bg-label-info">
                                <i class="bx bx-edit bx-lg"></i>
                            </span>
                        </div>
                        <h4 class="mb-0" id="updatedCount">0</h4>
                    </div>
                    <p class="mb-0">{{ __('Updated Records') }}</p>
                    <small class="text-muted">{{ __('Last 30 days') }}</small>
                </div>
            </div>
        </div>
        <div class="col-lg-3 col-sm-6 mb-6">
            <div class="card h-100">
                <div class="card-body">
                    <div class="d-flex align-items-center mb-2">
                        <div class="avatar me-4">
                            <span class="avatar-initial rounded bg-label-danger">
                                <i class="bx bx-trash bx-lg"></i>
                            </span>
                        </div>
                        <h4 class="mb-0" id="deletedCount">0</h4>
                    </div>
                    <p class="mb-0">{{ __('Deleted Records') }}</p>
                    <small class="text-muted">{{ __('Last 30 days') }}</small>
                </div>
            </div>
        </div>
    </div>

    <!-- Charts Row -->
    <div class="row mb-6">
        <div class="col-lg-8 mb-6">
            <div class="card h-100">
                <div class="card-header">
                    <h5 class="card-title mb-0">{{ __('Daily Activity Trend') }}</h5>
                </div>
                <div class="card-body">
                    <div id="dailyTrendChart"></div>
                </div>
            </div>
        </div>
        <div class="col-lg-4 mb-6">
            <div class="card h-100">
                <div class="card-header">
                    <h5 class="card-title mb-0">{{ __('Most Active Users') }}</h5>
                </div>
                <div class="card-body">
                    <div id="activeUsersChart"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Audit Logs Table -->
    <div class="card">
        <div class="card-header">
            <div class="d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">{{ __('Audit Logs') }}</h5>
                <button class="btn btn-label-primary" type="button" data-bs-toggle="collapse" data-bs-target="#filtersCollapse">
                    <i class="bx bx-filter-alt me-2"></i>{{ __('Filters') }}
                </button>
            </div>
        </div>

        <!-- Filters -->
        <div class="collapse" id="filtersCollapse">
            <div class="card-body border-top">
                <form id="filterForm">
                    <div class="row g-6">
                        <div class="col-md-3">
                            <label class="form-label">{{ __('Model') }}</label>
                            <select class="form-select select2" id="filter_auditable_type" name="auditable_type">
                                <option value="">{{ __('All Models') }}</option>
                            </select>
                        </div>
                        <div class="col-md-3">
                            <label class="form-label">{{ __('Event') }}</label>
                            <select class="form-select select2" id="filter_event" name="event">
                                <option value="">{{ __('All Events') }}</option>
                            </select>
                        </div>
                        <div class="col-md-3">
                            <label class="form-label">{{ __('User') }}</label>
                            <select class="form-select select2" id="filter_user_id" name="user_id">
                                <option value="">{{ __('All Users') }}</option>
                            </select>
                        </div>
                        <div class="col-md-3">
                            <label class="form-label">{{ __('Date Range') }}</label>
                            <input type="text" class="form-control" id="filter_date_range" placeholder="{{ __('Select date range') }}">
                            <input type="hidden" id="filter_date_from" name="date_from">
                            <input type="hidden" id="filter_date_to" name="date_to">
                        </div>
                        <div class="col-12">
                            <button type="button" class="btn btn-primary me-3" onclick="applyFilters()">
                                <i class="bx bx-search me-2"></i>{{ __('Apply Filters') }}
                            </button>
                            <button type="button" class="btn btn-label-secondary" onclick="clearFilters()">
                                <i class="bx bx-reset me-2"></i>{{ __('Clear') }}
                            </button>
                        </div>
                    </div>
                </form>
            </div>
        </div>

        <div class="card-datatable table-responsive">
            <table class="table" id="auditLogsTable">
                <thead>
                    <tr>
                        <th>{{ __('User') }}</th>
                        <th>{{ __('Model') }}</th>
                        <th>{{ __('Event') }}</th>
                        <th>{{ __('IP Address') }}</th>
                        <th>{{ __('Date & Time') }}</th>
                        <th>{{ __('Actions') }}</th>
                    </tr>
                </thead>
            </table>
        </div>
    </div>

    <!-- Audit Details Offcanvas -->
    <div class="offcanvas offcanvas-end" tabindex="-1" id="auditDetailsOffcanvas" style="width: 600px;">
        <div class="offcanvas-header">
            <h5 class="offcanvas-title">{{ __('Audit Log Details') }}</h5>
            <button type="button" class="btn-close text-reset" data-bs-dismiss="offcanvas"></button>
        </div>
        <div class="offcanvas-body" id="auditDetailsContent">
            <!-- Content will be loaded here -->
        </div>
    </div>
@endsection

@section('page-script')
    @vite(['Modules/AuditLog/resources/assets/js/audit-log.js'])
    <script>
        // Set pageData with translated labels
        const pageData = @json($pageData);
        pageData.labels = {
            ...pageData.labels,
            allModels: @json(__('All Models')),
            allEvents: @json(__('All Events')),
            allUsers: @json(__('All Users')),
            auditLogs: @json(__('Audit Logs')),
            numberOfLogs: @json(__('Number of Logs'))
        };
    </script>
@endsection
