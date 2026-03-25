@extends('layouts.layoutMaster')

@section('title', __('Resource Schedule') . ' - ' . $user->name)

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/fullcalendar/fullcalendar.scss'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.scss'])
    @vite(['resources/assets/vendor/libs/select2/select2.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/fullcalendar/fullcalendar.js'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.js'])
    @vite(['resources/assets/vendor/libs/select2/select2.js'])
@endsection

@section('page-script')
    <script>
        window.pageData = {
            userId: {{ $user->id }},
            userName: @json($user->name),
            allocations: @json($allocations),
            capacities: @json($capacities),
            urls: {
                updateAllocation: @json(route('pmcore.resources.update', ':id')),
                deleteAllocation: @json(route('pmcore.resources.destroy', ':id')),
                createAllocation: @json(route('pmcore.resources.store')),
                projectSearch: @json(route('pmcore.projects.search'))
            }
        };
    </script>
    @vite(['Modules/PMCore/resources/assets/js/resource-schedule.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Resource Schedule')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Resources'), 'url' => route('pmcore.resources.index')],
        ['name' => $user->name, 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<div class="row">
    <!-- Resource Info Card -->
    <div class="col-md-4 col-lg-3">
        <div class="card mb-4">
            <div class="card-body">
                <div class="d-flex align-items-center mb-3">
                    @if($user->profile_picture)
                        <img src="{{ $user->getProfilePicture() }}" alt="Avatar" class="rounded-circle me-3" width="60" height="60" />
                    @else
                        <div class="avatar avatar-xl me-3">
                            <span class="avatar-initial rounded-circle bg-label-primary">{{ $user->getInitials() }}</span>
                        </div>
                    @endif
                    <div>
                        <h5 class="mb-1">{{ $user->name }}</h5>
                        <small class="text-muted">{{ $user->email }}</small>
                    </div>
                </div>

                <div class="mb-3">
                    <label class="small text-muted d-block mb-1">{{ __('Department') }}</label>
                    <p class="mb-0">{{ $user->department->name ?? '-' }}</p>
                </div>

                <div class="mb-3">
                    <label class="small text-muted d-block mb-1">{{ __('Current Allocation') }}</label>
                    @php
                        $totalAllocation = $allocations->sum('allocation_percentage');
                        $statusClass = $totalAllocation > 100 ? 'danger' : ($totalAllocation >= 80 ? 'warning' : 'success');
                    @endphp
                    <div class="progress mb-2" style="height: 20px;">
                        <div class="progress-bar bg-{{ $statusClass }}" style="width: {{ min($totalAllocation, 100) }}%">
                            {{ $totalAllocation }}%
                        </div>
                    </div>
                </div>

                <div class="d-grid">
                    <button class="btn btn-primary" onclick="showAllocationForm()">
                        <i class="bx bx-plus me-1"></i>{{ __('New Allocation') }}
                    </button>
                </div>
            </div>
        </div>

        <!-- Current Allocations -->
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">{{ __('Active Allocations') }}</h6>
            </div>
            <div class="card-body">
                @if($allocations->count() > 0)
                    <div class="list-group list-group-flush">
                        @foreach($allocations as $allocation)
                            <div class="list-group-item px-0">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div class="flex-grow-1">
                                        <h6 class="mb-1">
                                            <a href="{{ route('pmcore.projects.show', $allocation->project_id) }}">
                                                {{ $allocation->project->name }}
                                            </a>
                                        </h6>
                                        <div class="small text-muted">
                                            <i class="bx bx-calendar me-1"></i>
                                            {{ $allocation->start_date->format('M d, Y') }}
                                            @if($allocation->end_date)
                                                - {{ $allocation->end_date->format('M d, Y') }}
                                            @else
                                                - {{ __('Ongoing') }}
                                            @endif
                                        </div>
                                        <div class="mt-1">
                                            <span class="badge bg-label-primary">{{ $allocation->allocation_percentage }}%</span>
                                            <span class="badge bg-label-{{ $allocation->status === 'active' ? 'success' : 'warning' }}">
                                                {{ ucfirst($allocation->status) }}
                                            </span>
                                        </div>
                                    </div>
                                    <div class="dropdown">
                                        <button class="btn btn-sm btn-icon" data-bs-toggle="dropdown">
                                            <i class="bx bx-dots-vertical-rounded"></i>
                                        </button>
                                        <ul class="dropdown-menu">
                                            <li>
                                                <a class="dropdown-item" href="{{ route('pmcore.resources.edit', $allocation->id) }}">
                                                    <i class="bx bx-edit me-1"></i>{{ __('Edit') }}
                                                </a>
                                            </li>
                                            <li>
                                                <a class="dropdown-item text-danger" href="#" onclick="deleteAllocation({{ $allocation->id }})">
                                                    <i class="bx bx-trash me-1"></i>{{ __('Delete') }}
                                                </a>
                                            </li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        @endforeach
                    </div>
                @else
                    <p class="text-muted text-center mb-0">{{ __('No active allocations') }}</p>
                @endif
            </div>
        </div>
    </div>

    <!-- Calendar View -->
    <div class="col-md-8 col-lg-9">
        <div class="card">
            <div class="card-header">
                <div class="d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">{{ __('Resource Calendar') }}</h5>
                    <div class="btn-group btn-group-sm" role="group">
                        <button type="button" class="btn btn-outline-primary" onclick="changeView('month')">{{ __('Month') }}</button>
                        <button type="button" class="btn btn-outline-primary" onclick="changeView('week')">{{ __('Week') }}</button>
                        <button type="button" class="btn btn-outline-primary" onclick="changeView('list')">{{ __('List') }}</button>
                    </div>
                </div>
            </div>
            <div class="card-body">
                <div id="resourceCalendar"></div>
            </div>
        </div>

        <!-- Legend -->
        <div class="card mt-3">
            <div class="card-body">
                <h6 class="mb-3">{{ __('Legend') }}</h6>
                <div class="d-flex flex-wrap gap-3">
                    <div class="d-flex align-items-center">
                        <div class="bg-primary rounded" style="width: 20px; height: 20px;"></div>
                        <span class="ms-2 small">{{ __('Project Allocation') }}</span>
                    </div>
                    <div class="d-flex align-items-center">
                        <div class="bg-success rounded" style="width: 20px; height: 20px;"></div>
                        <span class="ms-2 small">{{ __('Available') }}</span>
                    </div>
                    <div class="d-flex align-items-center">
                        <div class="bg-warning rounded" style="width: 20px; height: 20px;"></div>
                        <span class="ms-2 small">{{ __('Partial Allocation') }}</span>
                    </div>
                    <div class="d-flex align-items-center">
                        <div class="bg-danger rounded" style="width: 20px; height: 20px;"></div>
                        <span class="ms-2 small">{{ __('Overallocated') }}</span>
                    </div>
                    <div class="d-flex align-items-center">
                        <div class="bg-secondary rounded" style="width: 20px; height: 20px;"></div>
                        <span class="ms-2 small">{{ __('Non-working Day') }}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Allocation Form Modal -->
<div class="modal fade" id="allocationModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">{{ __('New Resource Allocation') }}</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <form id="allocationForm">
                <div class="modal-body">
                    <input type="hidden" name="user_id" value="{{ $user->id }}">

                    <div class="mb-3">
                        <label for="project_id" class="form-label">{{ __('Project') }} <span class="text-danger">*</span></label>
                        <select class="form-select" id="project_id" name="project_id" required>
                            <option value="">{{ __('Select Project') }}</option>
                        </select>
                        <div class="invalid-feedback"></div>
                    </div>

                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="start_date" class="form-label">{{ __('Start Date') }} <span class="text-danger">*</span></label>
                            <input type="text" class="form-control" id="start_date" name="start_date" placeholder="YYYY-MM-DD" required>
                            <div class="invalid-feedback"></div>
                        </div>
                        <div class="col-md-6">
                            <label for="end_date" class="form-label">{{ __('End Date') }}</label>
                            <input type="text" class="form-control" id="end_date" name="end_date" placeholder="YYYY-MM-DD">
                            <div class="invalid-feedback"></div>
                        </div>
                    </div>

                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="allocation_percentage" class="form-label">{{ __('Allocation %') }} <span class="text-danger">*</span></label>
                            <input type="number" class="form-control" id="allocation_percentage" name="allocation_percentage" min="0" max="100" value="100" required>
                            <div class="invalid-feedback"></div>
                        </div>
                        <div class="col-md-6">
                            <label for="hours_per_day" class="form-label">{{ __('Hours/Day') }} <span class="text-danger">*</span></label>
                            <input type="number" class="form-control" id="hours_per_day" name="hours_per_day" min="0.5" max="24" step="0.5" value="8" required>
                            <div class="invalid-feedback"></div>
                        </div>
                    </div>

                    <div class="mb-3">
                        <label for="allocation_type" class="form-label">{{ __('Allocation Type') }} <span class="text-danger">*</span></label>
                        <select class="form-select" id="allocation_type" name="allocation_type" required>
                            <option value="project" selected>{{ __('Entire Project') }}</option>
                            <option value="phase">{{ __('Project Phase') }}</option>
                            <option value="task">{{ __('Specific Task') }}</option>
                        </select>
                        <div class="invalid-feedback"></div>
                    </div>

                    <div class="mb-3" id="phase_section" style="display: none;">
                        <label for="phase" class="form-label">{{ __('Phase Name') }} <span class="text-danger">*</span></label>
                        <input type="text" class="form-control" id="phase" name="phase">
                        <div class="invalid-feedback"></div>
                    </div>

                    <div class="mb-3" id="task_section" style="display: none;">
                        <label for="task_id" class="form-label">{{ __('Task') }} <span class="text-danger">*</span></label>
                        <select class="form-select" id="task_id" name="task_id">
                            <option value="">{{ __('Select Task') }}</option>
                        </select>
                        <div class="invalid-feedback"></div>
                    </div>

                    <div class="mb-3">
                        <label for="notes" class="form-label">{{ __('Notes') }}</label>
                        <textarea class="form-control" id="notes" name="notes" rows="2"></textarea>
                        <div class="invalid-feedback"></div>
                    </div>

                    <div class="form-check form-switch mb-3">
                        <input class="form-check-input" type="checkbox" id="is_billable" name="is_billable" checked>
                        <label class="form-check-label" for="is_billable">
                            {{ __('Billable') }}
                        </label>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-label-secondary" data-bs-dismiss="modal">{{ __('Cancel') }}</button>
                    <button type="submit" class="btn btn-primary">{{ __('Create Allocation') }}</button>
                </div>
            </form>
        </div>
    </div>
</div>

@endsection
