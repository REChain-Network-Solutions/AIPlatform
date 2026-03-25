<!-- Stats Cards -->
<div class="row mb-4">
    <!-- First Row: Project Counts -->
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
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
                        <h5 class="mb-0">{{ $stats['total_projects'] ?? 0 }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
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
                        <h5 class="mb-0">{{ $stats['active_projects'] ?? 0 }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
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
                        <h5 class="mb-0">{{ $stats['completed_projects'] ?? 0 }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
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
                        <h5 class="mb-0">{{ $stats['overdue_projects'] ?? 0 }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Second Row: Task Stats -->
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-label-success rounded">
                            <i class="bx bx-task text-success"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Completed Tasks') }}</div>
                        <h5 class="mb-0">{{ $stats['completed_tasks'] ?? 0 }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-label-warning rounded">
                            <i class="bx bx-time-five text-warning"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Overdue Tasks') }}</div>
                        <h5 class="mb-0">{{ $stats['overdue_tasks'] ?? 0 }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-label-info rounded">
                            <i class="bx bx-dollar-circle text-info"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Budget Variance') }}</div>
                        <h5 class="mb-0 {{ ($stats['total_budget'] - $stats['total_spent']) >= 0 ? 'text-success' : 'text-danger' }}">
                            {{ \App\Helpers\FormattingHelper::formatCurrency(($stats['total_budget'] ?? 0) - ($stats['total_spent'] ?? 0)) }}
                        </h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-label-primary rounded">
                            <i class="bx bx-trending-up text-primary"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Profit Margin') }}</div>
                        <h5 class="mb-0 {{ ($stats['total_revenue'] - $stats['total_spent']) >= 0 ? 'text-success' : 'text-danger' }}">
                            {{ \App\Helpers\FormattingHelper::formatCurrency(($stats['total_revenue'] ?? 0) - ($stats['total_spent'] ?? 0)) }}
                        </h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Third Row: Budget Stats -->
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-label-primary rounded">
                            <i class="bx bx-wallet text-primary"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Total Budget') }}</div>
                        <h5 class="mb-0">{{ \App\Helpers\FormattingHelper::formatCurrency($stats['total_budget'] ?? 0) }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-label-warning rounded">
                            <i class="bx bx-credit-card text-warning"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Total Spent') }}</div>
                        <h5 class="mb-0">{{ \App\Helpers\FormattingHelper::formatCurrency($stats['total_spent'] ?? 0) }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-label-success rounded">
                            <i class="bx bx-trending-up text-success"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Total Revenue') }}</div>
                        <h5 class="mb-0">{{ \App\Helpers\FormattingHelper::formatCurrency($stats['total_revenue'] ?? 0) }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-lg-3 col-md-6 col-sm-6 mb-4">
        <div class="card">
            <div class="card-body">
                <div class="d-flex align-items-center">
                    <div class="avatar">
                        <div class="avatar-initial bg-label-danger rounded">
                            <i class="bx bx-error-circle text-danger"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Over Budget') }}</div>
                        <h5 class="mb-0">{{ $stats['projects_over_budget'] ?? 0 }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>