@extends('layouts.layoutMaster')

@section('title', __('Allocate Resource'))

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
        window.pageData = {
            project: @json($project ?? null),
            user: @json($user ?? null),
            urls: {
                store: @json(route('pmcore.resources.store')),
                index: @json(route('pmcore.resources.index')),
                projectSearch: @json(route('pmcore.projects.search')),
                userSearch: @json(route('pmcore.users.search')),
                availability: @json(route('pmcore.resources.availability')),
                projectTasks: '/pmcore/projects/:id/tasks'
            },
            labels: {
                success: @json(__('Success!')),
                error: @json(__('Error!'))
            }
        };
    </script>
    @vite(['Modules/PMCore/resources/assets/js/resource-create.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Allocate Resource')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Resources'), 'url' => route('pmcore.resources.index')],
        ['name' => __('Allocate'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<div class="row">
    <div class="col-md-8 col-lg-6">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">{{ __('Allocate Resource') }}</h5>
            </div>
            <div class="card-body">
                <form id="createAllocationForm">
                    <div class="mb-3">
                        <label for="user_id" class="form-label">{{ __('Resource') }} <span class="text-danger">*</span></label>
                        @if($user)
                            <input type="hidden" name="user_id" value="{{ $user->id }}">
                            <div class="form-control-static">
                                <x-datatable-user :user="$user" />
                            </div>
                        @else
                            <select class="form-select" id="user_id" name="user_id" required>
                                <option value="">{{ __('Select Resource') }}</option>
                            </select>
                            <div class="invalid-feedback"></div>
                        @endif
                    </div>

                    <div class="mb-3">
                        <label for="project_id" class="form-label">{{ __('Project') }} <span class="text-danger">*</span></label>
                        @if($project)
                            <input type="hidden" name="project_id" value="{{ $project->id }}">
                            <div class="form-control-static">
                                {{ $project->name }} ({{ $project->code }})
                            </div>
                        @else
                            <select class="form-select" id="project_id" name="project_id" required>
                                <option value="">{{ __('Select Project') }}</option>
                            </select>
                            <div class="invalid-feedback"></div>
                        @endif
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
                            <input type="number" class="form-control" id="allocation_percentage" name="allocation_percentage"
                                   min="0" max="100" value="100" required>
                            <div class="invalid-feedback"></div>
                        </div>
                        <div class="col-md-6">
                            <label for="hours_per_day" class="form-label">{{ __('Hours/Day') }} <span class="text-danger">*</span></label>
                            <input type="number" class="form-control" id="hours_per_day" name="hours_per_day"
                                   min="0.5" max="24" step="0.5" value="8" required>
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
                        <textarea class="form-control" id="notes" name="notes" rows="3"></textarea>
                        <div class="invalid-feedback"></div>
                    </div>

                    <div class="row mb-3">
                        <div class="col-md-6">
                            <div class="form-check form-switch">
                                <input class="form-check-input" type="checkbox" id="is_billable" name="is_billable" checked>
                                <label class="form-check-label" for="is_billable">
                                    {{ __('Billable') }}
                                </label>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="form-check form-switch">
                                <input class="form-check-input" type="checkbox" id="is_confirmed" name="is_confirmed">
                                <label class="form-check-label" for="is_confirmed">
                                    {{ __('Confirmed Allocation') }}
                                </label>
                            </div>
                        </div>
                    </div>

                    <div id="availability_preview" class="mb-3"></div>

                    <div class="d-flex gap-2">
                        <button type="submit" class="btn btn-primary">{{ __('Allocate Resource') }}</button>
                        <a href="{{ route('pmcore.resources.index') }}" class="btn btn-label-secondary">{{ __('Cancel') }}</a>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <!-- Availability Calendar Preview -->
    <div class="col-md-4 col-lg-6">
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">{{ __('Availability Information') }}</h6>
            </div>
            <div class="card-body">
                <div id="availability_info" class="text-center py-5">
                    <i class="bx bx-calendar bx-lg text-muted"></i>
                    <p class="text-muted mt-2">{{ __('Select a resource and date range to view availability') }}</p>
                </div>
            </div>
        </div>
    </div>
</div>

@endsection
