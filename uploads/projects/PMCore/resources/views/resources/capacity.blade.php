@extends('layouts.layoutMaster')

@section('title', __('Capacity Planning'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/apex-charts/apex-charts.scss'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.scss'])
    @vite(['resources/assets/vendor/libs/select2/select2.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/apex-charts/apexcharts.js'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.js'])
    @vite(['resources/assets/vendor/libs/select2/select2.js'])
@endsection

@section('page-script')
    <script>
        window.pageData = {
            capacityData: @json($capacityData),
            startDate: @json($startDate),
            endDate: @json($endDate),
            urls: {
                capacityData: @json(route('pmcore.resources.capacity.data')),
                capacity: @json(route('pmcore.resources.capacity'))
            },
            labels: {
                allocated: @json(__('Allocated')),
                available: @json(__('Available')),
                overallocated: @json(__('Overallocated')),
                utilization: @json(__('Utilization')),
                forecast: @json(__('Forecast')),
                heatmap: @json(__('Heatmap'))
            }
        };
    </script>
    @vite(['Modules/PMCore/resources/assets/js/capacity-planning.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Capacity Planning')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Resources'), 'url' => route('pmcore.resources.index')],
        ['name' => __('Capacity Planning'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<!-- Filters -->
<div class="card mb-4">
    <div class="card-body">
        <form id="capacityFilterForm" method="GET" action="{{ route('pmcore.resources.capacity') }}">
            <div class="row g-3">
                <div class="col-md-3">
                    <label for="start_date" class="form-label">{{ __('Start Date') }}</label>
                    <input type="text" class="form-control" id="start_date" name="start_date" value="{{ $startDate }}">
                </div>
                <div class="col-md-3">
                    <label for="end_date" class="form-label">{{ __('End Date') }}</label>
                    <input type="text" class="form-control" id="end_date" name="end_date" value="{{ $endDate }}">
                </div>
                <div class="col-md-3">
                    <label for="department_id" class="form-label">{{ __('Department') }}</label>
                    <select class="form-select" id="department_id" name="department_id">
                        <option value="">{{ __('All Departments') }}</option>
                        @foreach($departments as $department)
                            <option value="{{ $department->id }}" {{ request('department_id') == $department->id ? 'selected' : '' }}>
                                {{ $department->name }}
                            </option>
                        @endforeach
                    </select>
                </div>
                <div class="col-md-3">
                    <label class="form-label">&nbsp;</label>
                    <div>
                        <button type="submit" class="btn btn-primary">
                            <i class="bx bx-filter-alt me-1"></i>{{ __('Apply Filters') }}
                        </button>
                        <a href="{{ route('pmcore.resources.capacity') }}" class="btn btn-label-secondary">
                            {{ __('Reset') }}
                        </a>
                    </div>
                </div>
            </div>
        </form>
    </div>
</div>

<!-- Summary Stats -->
<div class="row mb-4">
    <div class="col-lg-3 col-md-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-primary rounded">
                            <i class="bx bx-group text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Total Resources') }}</div>
                        <h5 class="card-title mb-0">{{ $capacityData['total_resources'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-info rounded">
                            <i class="bx bx-pie-chart-alt-2 text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Overall Utilization') }}</div>
                        <h5 class="card-title mb-0">{{ $capacityData['utilization_percentage'] }}%</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-danger rounded">
                            <i class="bx bx-error text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Overallocated') }}</div>
                        <h5 class="card-title mb-0">{{ $capacityData['overallocated_resources'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-warning rounded">
                            <i class="bx bx-user-minus text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Underutilized') }}</div>
                        <h5 class="card-title mb-0">{{ $capacityData['underutilized_resources'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- View Toggle -->
<div class="card mb-4">
    <div class="card-header">
        <div class="d-flex justify-content-between align-items-center">
            <h5 class="mb-0">{{ __('Resource Capacity Analysis') }}</h5>
            <div class="btn-group" role="group">
                <button type="button" class="btn btn-sm btn-outline-primary active" data-view="utilization">
                    <i class="bx bx-bar-chart-alt-2 me-1"></i>{{ __('Utilization') }}
                </button>
                <button type="button" class="btn btn-sm btn-outline-primary" data-view="forecast">
                    <i class="bx bx-trending-up me-1"></i>{{ __('Forecast') }}
                </button>
                <button type="button" class="btn btn-sm btn-outline-primary" data-view="heatmap">
                    <i class="bx bx-grid-alt me-1"></i>{{ __('Heatmap') }}
                </button>
            </div>
        </div>
    </div>
    <div class="card-body">
        <div id="capacityChart"></div>
    </div>
</div>

<!-- Resource Breakdown Table -->
<div class="card">
    <div class="card-header">
        <h5 class="mb-0">{{ __('Resource Breakdown') }}</h5>
    </div>
    <div class="card-body">
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>{{ __('Resource') }}</th>
                        <th>{{ __('Department') }}</th>
                        <th>{{ __('Capacity Hours') }}</th>
                        <th>{{ __('Allocated Hours') }}</th>
                        <th>{{ __('Available Hours') }}</th>
                        <th>{{ __('Utilization') }}</th>
                        <th>{{ __('Status') }}</th>
                        <th>{{ __('Actions') }}</th>
                    </tr>
                </thead>
                <tbody>
                    @foreach($capacityData['resource_breakdown'] as $item)
                        <tr>
                            <td>
                                <div class="d-flex align-items-center">
                                    <div class="avatar avatar-sm me-3">
                                        @if($item['resource']->profile_picture)
                                            <img src="{{ $item['resource']->getProfilePicture() }}" alt="Avatar" class="rounded-circle" />
                                        @else
                                            <div class="avatar-initial bg-label-primary rounded-circle">
                                                {{ $item['resource']->getInitials() }}
                                            </div>
                                        @endif
                                    </div>
                                    <div>
                                        <h6 class="mb-0">{{ $item['resource']->name }}</h6>
                                        <small class="text-muted">{{ $item['resource']->email }}</small>
                                    </div>
                                </div>
                            </td>
                            <td>{{ $item['resource']->department->name ?? '-' }}</td>
                            <td>{{ number_format($item['capacity_hours']) }}h</td>
                            <td>{{ number_format($item['allocated_hours'], 1) }}h</td>
                            <td>{{ number_format($item['available_hours'], 1) }}h</td>
                            <td>
                                <div class="d-flex align-items-center">
                                    <div class="progress flex-grow-1 me-2" style="height: 10px;">
                                        <div class="progress-bar {{ $item['is_overallocated'] ? 'bg-danger' : ($item['is_underutilized'] ? 'bg-warning' : 'bg-success') }}"
                                             style="width: {{ min($item['utilization_percentage'], 100) }}%"></div>
                                    </div>
                                    <span class="small">{{ $item['utilization_percentage'] }}%</span>
                                </div>
                            </td>
                            <td>
                                @if($item['is_overallocated'])
                                    <span class="badge bg-label-danger">{{ __('Overallocated') }}</span>
                                @elseif($item['is_underutilized'])
                                    <span class="badge bg-label-warning">{{ __('Underutilized') }}</span>
                                @else
                                    <span class="badge bg-label-success">{{ __('Optimal') }}</span>
                                @endif
                            </td>
                            <td>
                                <a href="{{ route('pmcore.resources.schedule', $item['resource']->id) }}" class="btn btn-sm btn-icon btn-text-secondary">
                                    <i class="bx bx-calendar"></i>
                                </a>
                                <a href="{{ route('pmcore.resources.create', ['user_id' => $item['resource']->id]) }}" class="btn btn-sm btn-icon btn-text-secondary">
                                    <i class="bx bx-plus"></i>
                                </a>
                            </td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </div>
    </div>
</div>

@endsection
