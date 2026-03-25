@extends('layouts.layoutMaster')

@section('title', __('Project Reports'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/apex-charts/apex-charts.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/apex-charts/apexcharts.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Project Reports')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Reports'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<!-- Summary Cards -->
<div class="row mb-4">
    <div class="col-lg-3 col-sm-6 mb-4">
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

    <div class="col-lg-3 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-info rounded">
                            <i class="bx bx-trending-up text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Ongoing Projects') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['ongoing_projects'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="col-lg-3 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-success rounded">
                            <i class="bx bx-check-circle text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Completed Projects') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['completed_projects'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="col-lg-3 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-warning rounded">
                            <i class="bx bx-error text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Over Budget') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['projects_over_budget'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Financial Summary -->
<div class="row mb-4">
    <div class="col-lg-4 col-md-6 mb-4">
        <div class="card">
            <div class="card-body">
                <h6 class="card-title mb-3">{{ __('Total Budget') }}</h6>
                <h3 class="text-primary">{{ \App\Helpers\FormattingHelper::formatCurrency($stats['total_budget']) }}</h3>
                <small class="text-muted">{{ __('Across all active projects') }}</small>
            </div>
        </div>
    </div>

    <div class="col-lg-4 col-md-6 mb-4">
        <div class="card">
            <div class="card-body">
                <h6 class="card-title mb-3">{{ __('Total Spent') }}</h6>
                <h3 class="text-warning">{{ \App\Helpers\FormattingHelper::formatCurrency($stats['total_spent']) }}</h3>
                <small class="text-muted">{{ __('Actual costs incurred') }}</small>
            </div>
        </div>
    </div>

    <div class="col-lg-4 col-md-6 mb-4">
        <div class="card">
            <div class="card-body">
                <h6 class="card-title mb-3">{{ __('Total Revenue') }}</h6>
                <h3 class="text-success">{{ \App\Helpers\FormattingHelper::formatCurrency($stats['total_revenue']) }}</h3>
                <small class="text-muted">{{ __('From billable projects') }}</small>
            </div>
        </div>
    </div>
</div>

<!-- Report Links -->
<div class="row">
    <div class="col-lg-4 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center mb-3">
                    <div class="avatar">
                        <div class="avatar-initial bg-label-primary rounded">
                            <i class="bx bx-time-five"></i>
                        </div>
                    </div>
                    <h5 class="mb-0 ms-3">{{ __('Time Tracking Report') }}</h5>
                </div>
                <p class="text-muted mb-3">{{ __('View detailed time tracking data, billable hours, and time-based costs across projects.') }}</p>
                <a href="{{ route('pmcore.reports.time') }}" class="btn btn-primary">
                    <i class="bx bx-right-arrow-alt me-1"></i>{{ __('View Report') }}
                </a>
            </div>
        </div>
    </div>

    <div class="col-lg-4 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center mb-3">
                    <div class="avatar">
                        <div class="avatar-initial bg-label-success rounded">
                            <i class="bx bx-dollar"></i>
                        </div>
                    </div>
                    <h5 class="mb-0 ms-3">{{ __('Budget Report') }}</h5>
                </div>
                <p class="text-muted mb-3">{{ __('Track project budgets, actual costs, revenue, and profitability analysis.') }}</p>
                <a href="{{ route('pmcore.reports.budget') }}" class="btn btn-success">
                    <i class="bx bx-right-arrow-alt me-1"></i>{{ __('View Report') }}
                </a>
            </div>
        </div>
    </div>

    <div class="col-lg-4 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center mb-3">
                    <div class="avatar">
                        <div class="avatar-initial bg-label-info rounded">
                            <i class="bx bx-group"></i>
                        </div>
                    </div>
                    <h5 class="mb-0 ms-3">{{ __('Resource Utilization') }}</h5>
                </div>
                <p class="text-muted mb-3">{{ __('Analyze resource allocation, utilization rates, and capacity planning.') }}</p>
                <a href="{{ route('pmcore.reports.resource') }}" class="btn btn-info">
                    <i class="bx bx-right-arrow-alt me-1"></i>{{ __('View Report') }}
                </a>
            </div>
        </div>
    </div>
</div>

<!-- Average Completion Chart -->
<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">{{ __('Project Completion Overview') }}</h5>
            </div>
            <div class="card-body">
                <div id="completionChart"></div>
            </div>
        </div>
    </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    // Completion Chart
    const completionOptions = {
        series: [{{ $stats['average_completion'] ?? 0 }}],
        chart: {
            height: 350,
            type: 'radialBar',
            toolbar: {
                show: false
            }
        },
        plotOptions: {
            radialBar: {
                startAngle: -135,
                endAngle: 225,
                hollow: {
                    margin: 0,
                    size: '70%',
                    background: '#fff',
                    image: undefined,
                    imageOffsetX: 0,
                    imageOffsetY: 0,
                    position: 'front',
                    dropShadow: {
                        enabled: true,
                        top: 3,
                        left: 0,
                        blur: 4,
                        opacity: 0.24
                    }
                },
                track: {
                    background: '#fff',
                    strokeWidth: '67%',
                    margin: 0,
                    dropShadow: {
                        enabled: true,
                        top: -3,
                        left: 0,
                        blur: 4,
                        opacity: 0.35
                    }
                },
                dataLabels: {
                    show: true,
                    name: {
                        offsetY: -10,
                        show: true,
                        color: '#888',
                        fontSize: '17px'
                    },
                    value: {
                        formatter: function(val) {
                            return parseInt(val) + '%';
                        },
                        color: '#111',
                        fontSize: '36px',
                        show: true,
                    }
                }
            }
        },
        fill: {
            type: 'gradient',
            gradient: {
                shade: 'dark',
                type: 'horizontal',
                shadeIntensity: 0.5,
                gradientToColors: ['#ABE5A1'],
                inverseColors: true,
                opacityFrom: 1,
                opacityTo: 1,
                stops: [0, 100]
            }
        },
        stroke: {
            lineCap: 'round'
        },
        labels: ['{{ __("Average Completion") }}'],
    };

    const completionChart = new ApexCharts(document.querySelector("#completionChart"), completionOptions);
    completionChart.render();
});
</script>
@endsection
