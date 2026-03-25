@extends('layouts.layoutMaster')

@section('title', __('Time Tracking Report'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.scss'])
    @vite(['resources/assets/vendor/libs/select2/select2.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.js'])
    @vite(['resources/assets/vendor/libs/select2/select2.js'])
@endsection

@section('page-script')
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Initialize date pickers
    flatpickr('#start_date', {
        dateFormat: 'Y-m-d'
    });

    flatpickr('#end_date', {
        dateFormat: 'Y-m-d'
    });

    // Initialize select2
    if (typeof $ !== 'undefined') {
        $('#project_id, #user_id').select2({
            placeholder: 'Select...',
            allowClear: true
        });
    }
});

function exportReport() {
    // This will be handled by the DataImportExport addon
    alert('{{ __("Export functionality will be available with DataImportExport addon") }}');
}
</script>
@endsection

@section('content')
<x-breadcrumb
    :title="__('Time Tracking Report')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Reports'), 'url' => route('pmcore.reports.index')],
        ['name' => __('Time Report'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<!-- Filters -->
<div class="card mb-4">
    <div class="card-body">
        <form method="GET" action="{{ route('pmcore.reports.time') }}" class="row g-3">
            <div class="col-md-3">
                <label class="form-label">{{ __('Start Date') }}</label>
                <input type="text" class="form-control" id="start_date" name="start_date" value="{{ $startDate }}" placeholder="YYYY-MM-DD">
            </div>
            <div class="col-md-3">
                <label class="form-label">{{ __('End Date') }}</label>
                <input type="text" class="form-control" id="end_date" name="end_date" value="{{ $endDate }}" placeholder="YYYY-MM-DD">
            </div>
            <div class="col-md-3">
                <label class="form-label">{{ __('Project') }}</label>
                <select class="form-select" id="project_id" name="project_id">
                    <option value="">{{ __('All Projects') }}</option>
                    @foreach($projects as $project)
                        <option value="{{ $project->id }}" {{ $projectId == $project->id ? 'selected' : '' }}>
                            {{ $project->name }}
                        </option>
                    @endforeach
                </select>
            </div>
            <div class="col-md-3">
                <label class="form-label">{{ __('User') }}</label>
                <select class="form-select" id="user_id" name="user_id">
                    <option value="">{{ __('All Users') }}</option>
                    @foreach($users as $user)
                        <option value="{{ $user->id }}" {{ $userId == $user->id ? 'selected' : '' }}>
                            {{ $user->name }}
                        </option>
                    @endforeach
                </select>
            </div>
            <div class="col-12">
                <button type="submit" class="btn btn-primary">
                    <i class="bx bx-filter me-1"></i>{{ __('Apply Filters') }}
                </button>
                <a href="{{ route('pmcore.reports.time') }}" class="btn btn-outline-secondary">
                    <i class="bx bx-reset me-1"></i>{{ __('Reset') }}
                </a>
            </div>
        </form>
    </div>
</div>

<!-- Summary Cards -->
<div class="row mb-4">
    <div class="col-lg-2 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="small mb-1">{{ __('Total Hours') }}</div>
                <h4 class="mb-0">{{ number_format($summary['total_hours'], 2) }}</h4>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="small mb-1">{{ __('Billable Hours') }}</div>
                <h4 class="mb-0 text-success">{{ number_format($summary['billable_hours'], 2) }}</h4>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="small mb-1">{{ __('Non-Billable') }}</div>
                <h4 class="mb-0 text-warning">{{ number_format($summary['non_billable_hours'], 2) }}</h4>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="small mb-1">{{ __('Total Cost') }}</div>
                <h4 class="mb-0">{{ \App\Helpers\FormattingHelper::formatCurrency($summary['total_cost']) }}</h4>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="small mb-1">{{ __('Total Revenue') }}</div>
                <h4 class="mb-0 text-primary">{{ \App\Helpers\FormattingHelper::formatCurrency($summary['total_revenue']) }}</h4>
            </div>
        </div>
    </div>
</div>

<!-- Project Summary Table -->
<div class="card">
    <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="mb-0">{{ __('Project Summary') }}</h5>
        <button type="button" class="btn btn-outline-primary btn-sm" onclick="exportReport()">
            <i class="bx bx-download me-1"></i>{{ __('Export') }}
        </button>
    </div>
    <div class="table-responsive">
        <table class="table table-bordered">
            <thead>
                <tr>
                    <th>{{ __('Project') }}</th>
                    <th class="text-end">{{ __('Total Hours') }}</th>
                    <th class="text-end">{{ __('Billable Hours') }}</th>
                    <th class="text-end">{{ __('Cost') }}</th>
                    <th class="text-end">{{ __('Revenue') }}</th>
                    <th class="text-end">{{ __('Profit') }}</th>
                </tr>
            </thead>
            <tbody>
                @forelse($projectSummary as $item)
                    <tr>
                        <td>{{ $item->project->name ?? __('N/A') }}</td>
                        <td class="text-end">{{ number_format($item->total_hours, 2) }}</td>
                        <td class="text-end">{{ number_format($item->billable_hours, 2) }}</td>
                        <td class="text-end">{{ \App\Helpers\FormattingHelper::formatCurrency($item->total_cost) }}</td>
                        <td class="text-end">{{ \App\Helpers\FormattingHelper::formatCurrency($item->total_revenue) }}</td>
                        <td class="text-end">
                            @php
                                $profit = $item->total_revenue - $item->total_cost;
                            @endphp
                            <span class="{{ $profit >= 0 ? 'text-success' : 'text-danger' }}">
                                {{ \App\Helpers\FormattingHelper::formatCurrency($profit) }}
                            </span>
                        </td>
                    </tr>
                @empty
                    <tr>
                        <td colspan="6" class="text-center">{{ __('No data available') }}</td>
                    </tr>
                @endforelse
            </tbody>
            <tfoot>
                <tr>
                    <th>{{ __('Total') }}</th>
                    <th class="text-end">{{ number_format($summary['total_hours'], 2) }}</th>
                    <th class="text-end">{{ number_format($summary['billable_hours'], 2) }}</th>
                    <th class="text-end">{{ \App\Helpers\FormattingHelper::formatCurrency($summary['total_cost']) }}</th>
                    <th class="text-end">{{ \App\Helpers\FormattingHelper::formatCurrency($summary['total_revenue']) }}</th>
                    <th class="text-end">
                        @php
                            $totalProfit = $summary['total_revenue'] - $summary['total_cost'];
                        @endphp
                        <span class="{{ $totalProfit >= 0 ? 'text-success' : 'text-danger' }}">
                            {{ \App\Helpers\FormattingHelper::formatCurrency($totalProfit) }}
                        </span>
                    </th>
                </tr>
            </tfoot>
        </table>
    </div>
</div>
@endsection
