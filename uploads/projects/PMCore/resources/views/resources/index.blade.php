@extends('layouts.layoutMaster')

@section('title', __('Resource Management'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-responsive-bs5/responsive.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-buttons-bs5/buttons.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/select2/select2.scss'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables-bootstrap5.js'])
    @vite(['resources/assets/vendor/libs/select2/select2.js'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.js'])
@endsection

@section('page-script')
    <script>
        window.pageData = {
            urls: {
                datatable: @json(route('pmcore.resources.data')),
                create: @json(route('pmcore.resources.create')),
                schedule: @json(route('pmcore.resources.schedule', ':id')),
                availability: @json(route('pmcore.resources.availability')),
                projectSearch: @json(route('pmcore.projects.search'))
            },
            labels: {
                confirmDelete: @json(__('Are you sure you want to delete this allocation?')),
                success: @json(__('Success!')),
                error: @json(__('Error!'))
            }
        };
    </script>
    @vite(['Modules/PMCore/resources/assets/js/resources-list.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Resource Management')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Resources'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<!-- Statistics Cards -->
<div class="row mb-4">
    <div class="col-lg-3 col-md-6 col-sm-6">
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
                        <h5 class="card-title mb-0">{{ $stats['total_resources'] }}</h5>
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
                        <div class="avatar-initial bg-warning rounded">
                            <i class="bx bx-user-check text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Allocated') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['allocated_resources'] }}</h5>
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
                            <i class="bx bx-error text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Overallocated') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['overallocated_resources'] }}</h5>
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
                            <i class="bx bx-user-plus text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Available') }}</div>
                        <h5 class="card-title mb-0">{{ $stats['available_resources'] }}</h5>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Resources List -->
<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <div class="d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">{{ __('Resources') }}</h5>
                    <div class="d-flex gap-2">
                        <a href="{{ route('pmcore.resources.capacity') }}" class="btn btn-outline-primary">
                            <i class="bx bx-pie-chart-alt-2 me-1"></i>{{ __('Capacity Planning') }}
                        </a>
                        <a href="{{ route('pmcore.resources.create') }}" class="btn btn-primary">
                            <i class="bx bx-plus me-1"></i>{{ __('Allocate Resource') }}
                        </a>
                    </div>
                </div>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table id="resourcesTable" class="table table-bordered table-hover">
                        <thead>
                            <tr>
                                <th>{{ __('Resource') }}</th>
                                <th>{{ __('Role') }}</th>
                                <th>{{ __('Current Allocation') }}</th>
                                <th>{{ __('Availability') }}</th>
                                <th>{{ __('Actions') }}</th>
                            </tr>
                        </thead>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Allocate Resource Modal -->
<div class="offcanvas offcanvas-end" tabindex="-1" id="allocateResourceOffcanvas" style="width: 600px;">
    <div class="offcanvas-header">
        <h5 class="offcanvas-title">{{ __('Allocate Resource') }}</h5>
        <button type="button" class="btn-close text-reset" data-bs-dismiss="offcanvas"></button>
    </div>
    <div class="offcanvas-body">
        <form id="allocateResourceForm">
            <input type="hidden" id="resource_user_id" name="user_id">

            <div class="mb-3">
                <label class="form-label">{{ __('Resource') }}</label>
                <div id="resource_name" class="form-control-static"></div>
            </div>

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
                    <option value="project">{{ __('Entire Project') }}</option>
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
                <textarea class="form-control" id="notes" name="notes" rows="3"></textarea>
                <div class="invalid-feedback"></div>
            </div>

            <div class="mb-3">
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="is_billable" name="is_billable" checked>
                    <label class="form-check-label" for="is_billable">
                        {{ __('Billable') }}
                    </label>
                </div>
            </div>

            <div class="mb-3">
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="is_confirmed" name="is_confirmed">
                    <label class="form-check-label" for="is_confirmed">
                        {{ __('Confirmed Allocation') }}
                    </label>
                </div>
            </div>

            <div id="availability_preview" class="mb-3"></div>

            <div class="d-flex gap-2">
                <button type="submit" class="btn btn-primary flex-fill">{{ __('Allocate') }}</button>
                <button type="button" class="btn btn-label-secondary flex-fill" data-bs-dismiss="offcanvas">{{ __('Cancel') }}</button>
            </div>
        </form>
    </div>
</div>

@endsection
