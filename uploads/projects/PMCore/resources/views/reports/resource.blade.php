@extends('layouts.layoutMaster')

@section('title', __('Resource Utilization Report'))

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
        $('#department_id').select2({
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
    :title="__('Resource Utilization Report')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Reports'), 'url' => route('pmcore.reports.index')],
        ['name' => __('Resource Report'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<!-- Filters -->
<div class="card mb-4">
    <div class="card-body">
        <form method="GET" action="{{ route('pmcore.reports.resource') }}" class="row g-3">
            <div class="col-md-3">
                <label class="form-label">{{ __('Start Date') }}</label>
                <input type="text" class="form-control" id="start_date" name="start_date" value="{{ $startDate }}" placeholder="YYYY-MM-DD">
            </div>
            <div class="col-md-3">
                <label class="form-label">{{ __('End Date') }}</label>
                <input type="text" class="form-control" id="end_date" name="end_date" value="{{ $endDate }}" placeholder="YYYY-MM-DD">
            </div>
            <div class="col-md-3">
                <label class="form-label">{{ __('Department') }}</label>
                <select class="form-select" id="department_id" name="department_id">
                    <option value="">{{ __('All Departments') }}</option>
                    @foreach($departments as $department)
                        <option value="{{ $department->id }}" {{ $departmentId == $department->id ? 'selected' : '' }}>
                            {{ $department->name }}
                        </option>
                    @endforeach
                </select>
            </div>
            <div class="col-md-3 d-flex align-items-end">
                <button type="submit" class="btn btn-primary me-2">
                    <i class="bx bx-filter me-1"></i>{{ __('Apply Filters') }}
                </button>
                <a href="{{ route('pmcore.reports.resource') }}" class="btn btn-outline-secondary">
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
            <div class="card-body text-center">
                <h3 class="mb-1">{{ $summary['total_resources'] }}</h3>
                <small class="text-muted">{{ __('Total Resources') }}</small>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body text-center">
                <h3 class="mb-1">{{ number_format($summary['total_available_hours']) }}</h3>
                <small class="text-muted">{{ __('Available Hours') }}</small>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body text-center">
                <h3 class="mb-1">{{ number_format($summary['total_allocated_hours']) }}</h3>
                <small class="text-muted">{{ __('Allocated Hours') }}</small>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body text-center">
                <h3 class="mb-1">{{ number_format($summary['total_actual_hours']) }}</h3>
                <small class="text-muted">{{ __('Actual Hours') }}</small>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body text-center">
                <h3 class="mb-1 text-warning">{{ $summary['overallocated_resources'] }}</h3>
                <small class="text-muted">{{ __('Overallocated') }}</small>
            </div>
        </div>
    </div>
    <div class="col-lg-2 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body text-center">
                <h3 class="mb-1 text-info">{{ $summary['underutilized_resources'] }}</h3>
                <small class="text-muted">{{ __('Underutilized') }}</small>
            </div>
        </div>
    </div>
</div>

<!-- Average Utilization -->
<div class="card mb-4">
    <div class="card-body">
        <h6 class="card-title mb-3">{{ __('Average Resource Utilization') }}</h6>
        <div class="d-flex align-items-center">
            <div class="progress flex-grow-1" style="height: 30px;">
                @php
                    $avgUtilization = $summary['average_utilization'];
                    $utilizationClass = $avgUtilization > 100 ? 'bg-danger' : ($avgUtilization > 80 ? 'bg-warning' : 'bg-success');
                @endphp
                <div class="progress-bar {{ $utilizationClass }}" style="width: {{ min($avgUtilization, 100) }}%">
                    {{ round($avgUtilization, 1) }}%
                </div>
            </div>
            <div class="ms-3">
                <h4 class="mb-0">{{ round($avgUtilization, 1) }}%</h4>
            </div>
        </div>
        <div class="mt-2">
            <small class="text-muted">
                @if($avgUtilization < 70)
                    <i class="bx bx-info-circle"></i> {{ __('Resources are underutilized. Consider reassigning tasks or taking on more projects.') }}
                @elseif($avgUtilization > 100)
                    <i class="bx bx-error-circle"></i> {{ __('Resources are overallocated. Consider hiring more team members or extending deadlines.') }}
                @else
                    <i class="bx bx-check-circle"></i> {{ __('Resource utilization is at a healthy level.') }}
                @endif
            </small>
        </div>
    </div>
</div>

<!-- Resources Table -->
<div class="card">
    <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="mb-0">{{ __('Resource Details') }}</h5>
        <button type="button" class="btn btn-outline-primary btn-sm" onclick="exportReport()">
            <i class="bx bx-download me-1"></i>{{ __('Export') }}
        </button>
    </div>
    <div class="table-responsive">
        <table class="table table-hover">
            <thead>
                <tr>
                    <th>{{ __('Resource') }}</th>
                    <th>{{ __('Department') }}</th>
                    <th class="text-center">{{ __('Available Hours') }}</th>
                    <th class="text-center">{{ __('Allocated Hours') }}</th>
                    <th class="text-center">{{ __('Actual Hours') }}</th>
                    <th class="text-center">{{ __('Utilization %') }}</th>
                    <th>{{ __('Status') }}</th>
                    <th>{{ __('Allocation Bar') }}</th>
                </tr>
            </thead>
            <tbody>
                @forelse($resources as $resource)
                    <tr class="{{ $resource['is_overallocated'] ? 'table-warning' : ($resource['is_underutilized'] ? 'table-info' : '') }}">
                        <td>
                            <x-datatable-user :user="$resource['user']" />
                        </td>
                        <td>
                            @if($resource['user']->designation && $resource['user']->designation->department)
                                {{ $resource['user']->designation->department->name }}
                            @else
                                <span class="text-muted">-</span>
                            @endif
                        </td>
                        <td class="text-center">{{ number_format($resource['available_hours']) }}</td>
                        <td class="text-center">{{ number_format($resource['allocated_hours'], 1) }}</td>
                        <td class="text-center">{{ number_format($resource['actual_hours'], 1) }}</td>
                        <td class="text-center">
                            <span class="badge bg-{{ $resource['utilization_percentage'] > 100 ? 'danger' : ($resource['utilization_percentage'] > 80 ? 'warning' : 'success') }}">
                                {{ $resource['utilization_percentage'] }}%
                            </span>
                        </td>
                        <td>
                            @if($resource['is_overallocated'])
                                <span class="badge bg-danger">{{ __('Overallocated') }}</span>
                            @elseif($resource['is_underutilized'])
                                <span class="badge bg-info">{{ __('Underutilized') }}</span>
                            @else
                                <span class="badge bg-success">{{ __('Optimal') }}</span>
                            @endif
                        </td>
                        <td style="width: 200px;">
                            <div class="progress" style="height: 15px;">
                                @php
                                    $allocatedPercent = $resource['available_hours'] > 0 ? ($resource['allocated_hours'] / $resource['available_hours']) * 100 : 0;
                                    $actualPercent = $resource['available_hours'] > 0 ? ($resource['actual_hours'] / $resource['available_hours']) * 100 : 0;
                                @endphp
                                <div class="progress-bar bg-primary" style="width: {{ min($actualPercent, 100) }}%" title="{{ __('Actual Hours') }}"></div>
                                <div class="progress-bar bg-primary bg-opacity-50" style="width: {{ min(max($allocatedPercent - $actualPercent, 0), 100 - $actualPercent) }}%" title="{{ __('Allocated but not used') }}"></div>
                            </div>
                            <small class="text-muted">
                                {{ __('Allocated') }}: {{ round($allocatedPercent) }}% |
                                {{ __('Used') }}: {{ round($actualPercent) }}%
                            </small>
                        </td>
                    </tr>
                @empty
                    <tr>
                        <td colspan="8" class="text-center">{{ __('No resources found') }}</td>
                    </tr>
                @endforelse
            </tbody>
        </table>
    </div>
</div>
@endsection
