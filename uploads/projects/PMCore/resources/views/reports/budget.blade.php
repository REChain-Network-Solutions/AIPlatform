@extends('layouts.layoutMaster')

@section('title', __('Budget Report'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/select2/select2.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/select2/select2.js'])
@endsection

@section('page-script')
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Initialize select2
    if (typeof $ !== 'undefined') {
        $('#status, #type').select2({
            placeholder: 'Select...',
            allowClear: true
        });
    }
});

function exportReport() {
    alert('{{ __("Export functionality will be available with DataImportExport addon") }}');
}
</script>
@endsection

@section('content')
<x-breadcrumb
    :title="__('Budget Report')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Reports'), 'url' => route('pmcore.reports.index')],
        ['name' => __('Budget Report'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<!-- Filters -->
<div class="card mb-4">
    <div class="card-body">
        <form method="GET" action="{{ route('pmcore.reports.budget') }}" class="row g-3">
            <div class="col-md-4">
                <label class="form-label">{{ __('Status') }}</label>
                <select class="form-select" id="status" name="status">
                    <option value="">{{ __('All Statuses') }}</option>
                    @foreach(\Modules\PMCore\app\Enums\ProjectStatus::cases() as $statusOption)
                        <option value="{{ $statusOption->value }}" {{ $status == $statusOption->value ? 'selected' : '' }}>
                            {{ $statusOption->label() }}
                        </option>
                    @endforeach
                </select>
            </div>
            <div class="col-md-4">
                <label class="form-label">{{ __('Type') }}</label>
                <select class="form-select" id="type" name="type">
                    <option value="">{{ __('All Types') }}</option>
                    @foreach(\Modules\PMCore\app\Enums\ProjectType::cases() as $typeOption)
                        <option value="{{ $typeOption->value }}" {{ $type == $typeOption->value ? 'selected' : '' }}>
                            {{ $typeOption->label() }}
                        </option>
                    @endforeach
                </select>
            </div>
            <div class="col-md-4 d-flex align-items-end">
                <button type="submit" class="btn btn-primary me-2">
                    <i class="bx bx-filter me-1"></i>{{ __('Apply Filters') }}
                </button>
                <a href="{{ route('pmcore.reports.budget') }}" class="btn btn-outline-secondary">
                    <i class="bx bx-reset me-1"></i>{{ __('Reset') }}
                </a>
            </div>
        </form>
    </div>
</div>

<!-- Summary Cards -->
<div class="row mb-4">
    <div class="col-lg-3 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <div class="small mb-1">{{ __('Total Budget') }}</div>
                        <h5 class="mb-0">{{ \App\Helpers\FormattingHelper::formatCurrency($summary['total_budget']) }}</h5>
                    </div>
                    <div class="avatar">
                        <div class="avatar-initial bg-label-primary rounded">
                            <i class="bx bx-wallet"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <div class="small mb-1">{{ __('Total Cost') }}</div>
                        <h5 class="mb-0">{{ \App\Helpers\FormattingHelper::formatCurrency($summary['total_actual_cost']) }}</h5>
                    </div>
                    <div class="avatar">
                        <div class="avatar-initial bg-label-warning rounded">
                            <i class="bx bx-credit-card"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <div class="small mb-1">{{ __('Total Revenue') }}</div>
                        <h5 class="mb-0">{{ \App\Helpers\FormattingHelper::formatCurrency($summary['total_actual_revenue']) }}</h5>
                    </div>
                    <div class="avatar">
                        <div class="avatar-initial bg-label-success rounded">
                            <i class="bx bx-trending-up"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <div class="small mb-1">{{ __('Total Profit') }}</div>
                        <h5 class="mb-0 {{ $summary['total_profit'] >= 0 ? 'text-success' : 'text-danger' }}">
                            {{ \App\Helpers\FormattingHelper::formatCurrency($summary['total_profit']) }}
                        </h5>
                    </div>
                    <div class="avatar">
                        <div class="avatar-initial bg-label-info rounded">
                            <i class="bx bx-dollar"></i>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Project Status Summary -->
<div class="row mb-4">
    <div class="col-md-6">
        <div class="card h-100">
            <div class="card-body">
                <h6 class="card-title">{{ __('Budget Status Overview') }}</h6>
                <div class="row">
                    <div class="col-6">
                        <div class="d-flex align-items-center mb-3">
                            <div class="badge bg-success me-2">{{ $summary['projects_on_track'] }}</div>
                            <span>{{ __('Projects on Budget') }}</span>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="d-flex align-items-center mb-3">
                            <div class="badge bg-danger me-2">{{ $summary['projects_over_budget'] }}</div>
                            <span>{{ __('Projects Over Budget') }}</span>
                        </div>
                    </div>
                </div>
                <div class="progress mb-2" style="height: 20px;">
                    @php
                        $totalProjects = $summary['projects_on_track'] + $summary['projects_over_budget'];
                        $onTrackPercentage = $totalProjects > 0 ? ($summary['projects_on_track'] / $totalProjects) * 100 : 0;
                    @endphp
                    <div class="progress-bar bg-success" style="width: {{ $onTrackPercentage }}%">
                        {{ round($onTrackPercentage) }}%
                    </div>
                    <div class="progress-bar bg-danger" style="width: {{ 100 - $onTrackPercentage }}%">
                        {{ round(100 - $onTrackPercentage) }}%
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card h-100">
            <div class="card-body">
                <h6 class="card-title">{{ __('Financial Summary') }}</h6>
                <div class="table-responsive">
                    <table class="table table-sm">
                        <tr>
                            <td>{{ __('Total Budget') }}</td>
                            <td class="text-end">{{ \App\Helpers\FormattingHelper::formatCurrency($summary['total_budget']) }}</td>
                        </tr>
                        <tr>
                            <td>{{ __('Total Spent') }}</td>
                            <td class="text-end">{{ \App\Helpers\FormattingHelper::formatCurrency($summary['total_actual_cost']) }}</td>
                        </tr>
                        <tr>
                            <td>{{ __('Budget Remaining') }}</td>
                            <td class="text-end {{ ($summary['total_budget'] - $summary['total_actual_cost']) >= 0 ? 'text-success' : 'text-danger' }}">
                                {{ \App\Helpers\FormattingHelper::formatCurrency($summary['total_budget'] - $summary['total_actual_cost']) }}
                            </td>
                        </tr>
                        <tr>
                            <td>{{ __('Budget Utilization') }}</td>
                            <td class="text-end">
                                @php
                                    $utilization = $summary['total_budget'] > 0 ? ($summary['total_actual_cost'] / $summary['total_budget']) * 100 : 0;
                                @endphp
                                <span class="{{ $utilization > 100 ? 'text-danger' : 'text-success' }}">
                                    {{ round($utilization, 1) }}%
                                </span>
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Projects Table -->
<div class="card">
    <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="mb-0">{{ __('Project Budget Details') }}</h5>
        <button type="button" class="btn btn-outline-primary btn-sm" onclick="exportReport()">
            <i class="bx bx-download me-1"></i>{{ __('Export') }}
        </button>
    </div>
    <div class="table-responsive">
        <table class="table table-hover">
            <thead>
                <tr>
                    <th>{{ __('Project') }}</th>
                    <th>{{ __('Client') }}</th>
                    <th>{{ __('Status') }}</th>
                    <th class="text-end">{{ __('Budget') }}</th>
                    <th class="text-end">{{ __('Actual Cost') }}</th>
                    <th class="text-end">{{ __('Revenue') }}</th>
                    <th class="text-end">{{ __('Variance') }}</th>
                    <th class="text-end">{{ __('Profit') }}</th>
                    <th class="text-center">{{ __('Hours') }}</th>
                </tr>
            </thead>
            <tbody>
                @forelse($projects as $data)
                    <tr class="{{ $data['is_over_budget'] ? 'table-danger' : '' }}">
                        <td>
                            <a href="{{ route('pmcore.projects.show', $data['project']->id) }}">
                                {{ $data['project']->name }}
                            </a>
                        </td>
                        <td>{{ $data['project']->client?->name ?? '-' }}</td>
                        <td>
                            <span class="badge bg-{{ $data['project']->status->color() }}">
                                {{ $data['project']->status->label() }}
                            </span>
                        </td>
                        <td class="text-end">{{ \App\Helpers\FormattingHelper::formatCurrency($data['budget']) }}</td>
                        <td class="text-end">{{ \App\Helpers\FormattingHelper::formatCurrency($data['actual_cost']) }}</td>
                        <td class="text-end">{{ \App\Helpers\FormattingHelper::formatCurrency($data['actual_revenue']) }}</td>
                        <td class="text-end">
                            <span class="{{ $data['budget_variance'] >= 0 ? 'text-success' : 'text-danger' }}">
                                {{ \App\Helpers\FormattingHelper::formatCurrency($data['budget_variance']) }}
                                @if($data['budget'] > 0)
                                    <small>({{ $data['budget_variance_percentage'] }}%)</small>
                                @endif
                            </span>
                        </td>
                        <td class="text-end">
                            <span class="{{ $data['profit_margin'] >= 0 ? 'text-success' : 'text-danger' }}">
                                {{ \App\Helpers\FormattingHelper::formatCurrency($data['profit_margin']) }}
                                @if($data['actual_revenue'] > 0)
                                    <small>({{ $data['profit_margin_percentage'] }}%)</small>
                                @endif
                            </span>
                        </td>
                        <td class="text-center">
                            <small>
                                {{ number_format($data['total_hours'], 1) }} /
                                {{ number_format($data['billable_hours'], 1) }}
                            </small>
                        </td>
                    </tr>
                @empty
                    <tr>
                        <td colspan="9" class="text-center">{{ __('No projects found') }}</td>
                    </tr>
                @endforelse
            </tbody>
        </table>
    </div>
</div>
@endsection
