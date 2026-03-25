@extends('layouts.layoutMaster')

@section('title', __('Edit Timesheet'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/select2/select2.scss'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/select2/select2.js'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.js'])
@endsection

@section('page-script')
    <script>
        window.pageData = {
            urls: {
                updateUrl: @json(route('pmcore.timesheets.update', ['timesheet' => $timesheet->id])),
                projectTasksUrl: @json(route('pmcore.timesheets.project-tasks', ['project' => '__PROJECT_ID__'])),
                indexUrl: @json(route('pmcore.timesheets.index')),
                usersSearchUrl: @json(route('pmcore.users.search')),
                projectsSearchUrl: @json(route('pmcore.projects.search'))
            },
            labels: {
                selectProject: @json(__('Please select a project first')),
                loadingTasks: @json(__('Loading tasks...')),
                noTasksFound: @json(__('No tasks found for this project')),
                error: @json(__('An error occurred. Please try again.')),
                updateSuccess: @json(__('Timesheet updated successfully!'))
            },
            timesheet: {
                user_id: {{ $timesheet->user_id }},
                project_id: {{ $timesheet->project_id }},
                task_id: {{ $timesheet->task_id ?: 'null' }},
                is_billable: {{ $timesheet->is_billable ? 'true' : 'false' }}
            }
        };
    </script>
    @vite(['Modules/PMCore/resources/assets/js/timesheets-edit.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Edit Timesheet')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Timesheets'), 'url' => route('pmcore.timesheets.index')],
        ['name' => __('Edit'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<div class="row">
    <div class="col-lg-8 mx-auto">
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">{{ __('Edit Timesheet Entry') }}</h5>
                @if($timesheet->status !== \Modules\PMCore\app\Enums\TimesheetStatus::DRAFT)
                    <div class="alert alert-warning mt-3 mb-0">
                        <i class="bx bx-info-circle me-2"></i>
                        {{ __('This timesheet has been :status and cannot be edited.', ['status' => strtolower($timesheet->status->label())]) }}
                    </div>
                @endif
            </div>
            <div class="card-body">
                <form id="timesheetForm" method="POST">
                    @csrf
                    @method('PUT')

                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="user_id" class="form-label">{{ __('User') }} <span class="text-danger">*</span></label>
                            <select class="form-select" id="user_id" name="user_id" required>
                                <option value="{{ $timesheet->user_id }}" selected>{{ $timesheet->user->name }}</option>
                            </select>
                            <div class="invalid-feedback"></div>
                        </div>

                        <div class="col-md-6">
                            <label for="date" class="form-label">{{ __('Date') }} <span class="text-danger">*</span></label>
                            <input type="text" class="form-control" id="date" name="date" value="{{ $timesheet->date->format('Y-m-d') }}" placeholder="YYYY-MM-DD" required>
                            <div class="invalid-feedback"></div>
                        </div>
                    </div>

                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="project_id" class="form-label">{{ __('Project') }} <span class="text-danger">*</span></label>
                            <select class="form-select" id="project_id" name="project_id" required>
                                <option value="{{ $timesheet->project_id }}" selected>{{ $timesheet->project->name }}</option>
                            </select>
                            <div class="invalid-feedback"></div>
                        </div>

                        <div class="col-md-6">
                            <label for="task_id" class="form-label">{{ __('Task') }}</label>
                            <select class="form-select" id="task_id" name="task_id">
                                @if($timesheet->task)
                                    <option value="{{ $timesheet->task_id }}" selected>{{ $timesheet->task->title }}</option>
                                @else
                                    <option value="">{{ __('Select Task (Optional)') }}</option>
                                @endif
                            </select>
                            <div class="invalid-feedback"></div>
                        </div>
                    </div>

                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="hours" class="form-label">{{ __('Hours') }} <span class="text-danger">*</span></label>
                            <input type="number" class="form-control" id="hours" name="hours" value="{{ $timesheet->hours }}" step="0.01" min="0.01" max="24" required>
                            <small class="text-muted">{{ __('Enter time in hours (e.g., 1.5 for 1 hour 30 minutes)') }}</small>
                            <div class="invalid-feedback"></div>
                        </div>

                        <div class="col-md-6">
                            <label for="is_billable" class="form-label">{{ __('Billable') }}</label>
                            <div class="mt-2">
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="is_billable" name="is_billable" {{ $timesheet->is_billable ? 'checked' : '' }}>
                                    <label class="form-check-label" for="is_billable">
                                        {{ __('This time is billable to the client') }}
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="row mb-3" id="ratesSection" style="{{ $timesheet->is_billable ? '' : 'display: none;' }}">
                        <div class="col-md-6">
                            <label for="billing_rate" class="form-label">{{ __('Billing Rate') }}</label>
                            <div class="input-group">
                                <span class="input-group-text">$</span>
                                <input type="number" class="form-control" id="billing_rate" name="billing_rate" value="{{ $timesheet->billing_rate }}" step="0.01" min="0">
                                <span class="input-group-text">{{ __('/hour') }}</span>
                            </div>
                            <small class="text-muted">{{ __('Leave empty to use project default rate') }}</small>
                            <div class="invalid-feedback"></div>
                        </div>

                        <div class="col-md-6">
                            <label for="cost_rate" class="form-label">{{ __('Cost Rate') }}</label>
                            <div class="input-group">
                                <span class="input-group-text">$</span>
                                <input type="number" class="form-control" id="cost_rate" name="cost_rate" value="{{ $timesheet->cost_rate }}" step="0.01" min="0">
                                <span class="input-group-text">{{ __('/hour') }}</span>
                            </div>
                            <small class="text-muted">{{ __('Internal cost rate for this time') }}</small>
                            <div class="invalid-feedback"></div>
                        </div>
                    </div>

                    <div class="mb-3">
                        <label for="description" class="form-label">{{ __('Description') }}</label>
                        <textarea class="form-control" id="description" name="description" rows="4" placeholder="{{ __('Describe the work done...') }}">{{ $timesheet->description }}</textarea>
                        <div class="invalid-feedback"></div>
                    </div>

                    <div class="mb-3">
                        <div class="d-flex align-items-center">
                            <span class="me-3">{{ __('Status') }}:</span>
                            {!! $timesheet->status_badge !!}
                            @if($timesheet->approved_by_id)
                                <span class="ms-3 text-muted">
                                    {{ __('by :name on :date', [
                                        'name' => $timesheet->approvedBy->name,
                                        'date' => $timesheet->approved_at->format('M d, Y')
                                    ]) }}
                                </span>
                            @endif
                        </div>
                    </div>

                    @if($timesheet->canBeEditedBy(auth()->user()))
                        <div class="d-flex gap-2">
                            <button type="submit" class="btn btn-primary">
                                <i class="bx bx-save me-1"></i>{{ __('Update Timesheet') }}
                            </button>
                            <a href="{{ route('pmcore.timesheets.index') }}" class="btn btn-label-secondary">
                                <i class="bx bx-x me-1"></i>{{ __('Cancel') }}
                            </a>
                        </div>
                    @else
                        <a href="{{ route('pmcore.timesheets.index') }}" class="btn btn-label-secondary">
                            <i class="bx bx-arrow-back me-1"></i>{{ __('Back to List') }}
                        </a>
                    @endif
                </form>
            </div>
        </div>
    </div>
</div>

@endsection
