@extends('layouts.layoutMaster')

@section('title', __('Project Dashboard'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/apex-charts/apex-charts.scss'])
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-responsive-bs5/responsive.bootstrap5.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/apex-charts/apexcharts.js'])
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables-bootstrap5.js'])
@endsection

@section('page-script')
    @vite(['Modules/PMCore/resources/assets/js/dashboard.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Project Dashboard')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Dashboard'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<!-- Stats Cards -->
<div class="row mb-4">
    <div class="col-lg-3 col-md-6 col-sm-6">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-primary rounded">
                            <i class="bx bx-folder text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Total Projects') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['total_projects'] }}</h5>
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
                            <i class="bx bx-play-circle text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Active Projects') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['active_projects'] }}</h5>
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
                        <div class="avatar-initial bg-info rounded">
                            <i class="bx bx-check-circle text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Completed') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['completed_projects'] }}</h5>
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
                            <i class="bx bx-time text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Overdue') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['overdue_projects'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Charts Row -->
<div class="row mb-4">
    <!-- Project Status Distribution -->
    <div class="col-md-4">
        <div class="card">
            <div class="card-header d-flex justify-content-between">
                <h5 class="card-title mb-0">{{ __('Project Status Distribution') }}</h5>
            </div>
            <div class="card-body">
                <div id="projectStatusChart"></div>
            </div>
        </div>
    </div>

    <!-- Monthly Project Creation -->
    <div class="col-md-4">
        <div class="card">
            <div class="card-header d-flex justify-content-between">
                <h5 class="card-title mb-0">{{ __('Monthly Project Creation') }}</h5>
            </div>
            <div class="card-body">
                <div id="monthlyCreationChart"></div>
            </div>
        </div>
    </div>

    <!-- Project Completion Trend -->
    <div class="col-md-4">
        <div class="card">
            <div class="card-header d-flex justify-content-between">
                <h5 class="card-title mb-0">{{ __('Completion Trend') }}</h5>
            </div>
            <div class="card-body">
                <div id="completionTrendChart"></div>
            </div>
        </div>
    </div>
</div>

<!-- Recent Projects and Overdue Projects -->
<div class="row">
    <!-- Recent Projects -->
    <div class="col-md-6">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">{{ __('Recent Projects') }}</h5>
                <a href="{{ route('pmcore.projects.index') }}" class="btn btn-sm btn-outline-primary">{{ __('View All') }}</a>
            </div>
            <div class="card-body">
                @if($recentProjects->count() > 0)
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>{{ __('Project') }}</th>
                                    <th>{{ __('Client') }}</th>
                                    <th>{{ __('Status') }}</th>
                                    <th>{{ __('Created') }}</th>
                                </tr>
                            </thead>
                            <tbody>
                                @foreach($recentProjects as $project)
                                <tr>
                                    <td>
                                        <a href="{{ route('pmcore.projects.show', $project->id) }}" class="text-decoration-none">
                                            {{ $project->name }}
                                        </a>
                                    </td>
                                    <td>{{ $project->client?->name ?? '-' }}</td>
                                    <td>
                                        <span class="badge bg-label-{{ $project->status->color() }}">
                                            {{ $project->status->label() }}
                                        </span>
                                    </td>
                                    <td>{{ $project->created_at->format('M d, Y') }}</td>
                                </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                @else
                    <div class="text-center py-3">
                        <i class="bx bx-folder bx-lg text-muted"></i>
                        <p class="text-muted mt-2">{{ __('No recent projects') }}</p>
                    </div>
                @endif
            </div>
        </div>
    </div>

    <!-- Overdue Projects -->
    <div class="col-md-6">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">{{ __('Overdue Projects') }}</h5>
                <a href="{{ route('pmcore.projects.index') }}?filter=overdue" class="btn btn-sm btn-outline-danger">{{ __('View All') }}</a>
            </div>
            <div class="card-body">
                @if($overdueProjects->count() > 0)
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>{{ __('Project') }}</th>
                                    <th>{{ __('Client') }}</th>
                                    <th>{{ __('Due Date') }}</th>
                                    <th>{{ __('Days Over') }}</th>
                                </tr>
                            </thead>
                            <tbody>
                                @foreach($overdueProjects as $project)
                                <tr>
                                    <td>
                                        <a href="{{ route('pmcore.projects.show', $project->id) }}" class="text-decoration-none">
                                            {{ $project->name }}
                                        </a>
                                    </td>
                                    <td>{{ $project->client?->name ?? '-' }}</td>
                                    <td>{{ $project->end_date?->format('M d, Y') ?? '-' }}</td>
                                    <td>
                                        @if($project->end_date)
                                            <span class="badge bg-label-danger">
                                                {{ now()->diffInDays($project->end_date) }} {{ __('days') }}
                                            </span>
                                        @else
                                            -
                                        @endif
                                    </td>
                                </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                @else
                    <div class="text-center py-3">
                        <i class="bx bx-check-circle bx-lg text-success"></i>
                        <p class="text-muted mt-2">{{ __('No overdue projects') }}</p>
                    </div>
                @endif
            </div>
        </div>
    </div>
</div>

<script>
const dashboardData = {
    chartData: @json($chartData),
    currencySymbol: '$'
};
</script>
@endsection
