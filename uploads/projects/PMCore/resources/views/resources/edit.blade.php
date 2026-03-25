@extends('layouts.layoutMaster')

@section('title', __('Edit Resource Allocation'))

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
            allocation: @json($allocation),
            urls: {
                update: @json(route('pmcore.resources.update', $allocation->id)),
                index: @json(route('pmcore.resources.index')),
                projectSearch: @json(route('pmcore.projects.search')),
                projectTasks: '/pmcore/projects/:id/tasks'
            },
            labels: {
                success: @json(__('Success!')),
                error: @json(__('Error!'))
            }
        };
    </script>
    @vite(['Modules/PMCore/resources/assets/js/resource-edit.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Edit Resource Allocation')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Resources'), 'url' => route('pmcore.resources.index')],
        ['name' => __('Edit'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<div class="row">
    <div class="col-md-8 col-lg-6">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">{{ __('Edit Resource Allocation') }}</h5>
            </div>
            <div class="card-body">
                <form id="editAllocationForm">
                    <div class="mb-3">
                        <label class="form-label">{{ __('Resource') }}</label>
                        <div class="form-control-static">
                            <x-datatable-user :user="$allocation->user" />
                        </div>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">{{ __('Project') }}</label>
                        <div class="form-control-static">
                            {{ $allocation->project->name }} ({{ $allocation->project->code }})
                        </div>
                    </div>

                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="start_date" class="form-label">{{ __('Start Date') }} <span class="text-danger">*</span></label>
                            <input type="text" class="form-control" id="start_date" name="start_date"
                                   value="{{ $allocation->start_date->format('Y-m-d') }}" required>
                            <div class="invalid-feedback"></div>
                        </div>
                        <div class="col-md-6">
                            <label for="end_date" class="form-label">{{ __('End Date') }}</label>
                            <input type="text" class="form-control" id="end_date" name="end_date"
                                   value="{{ $allocation->end_date ? $allocation->end_date->format('Y-m-d') : '' }}">
                            <div class="invalid-feedback"></div>
                        </div>
                    </div>

                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="allocation_percentage" class="form-label">{{ __('Allocation %') }} <span class="text-danger">*</span></label>
                            <input type="number" class="form-control" id="allocation_percentage" name="allocation_percentage"
                                   min="0" max="100" value="{{ $allocation->allocation_percentage }}" required>
                            <div class="invalid-feedback"></div>
                        </div>
                        <div class="col-md-6">
                            <label for="hours_per_day" class="form-label">{{ __('Hours/Day') }} <span class="text-danger">*</span></label>
                            <input type="number" class="form-control" id="hours_per_day" name="hours_per_day"
                                   min="0.5" max="24" step="0.5" value="{{ $allocation->hours_per_day }}" required>
                            <div class="invalid-feedback"></div>
                        </div>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">{{ __('Allocation Type') }}</label>
                        <div class="form-control-static">
                            @if($allocation->allocation_type === 'project')
                                {{ __('Entire Project') }}
                            @elseif($allocation->allocation_type === 'phase')
                                {{ __('Project Phase') }}: {{ $allocation->phase }}
                            @elseif($allocation->allocation_type === 'task')
                                {{ __('Specific Task') }}: {{ $allocation->task->title ?? 'N/A' }}
                            @endif
                        </div>
                    </div>

                    <div class="mb-3">
                        <label for="status" class="form-label">{{ __('Status') }} <span class="text-danger">*</span></label>
                        <select class="form-select" id="status" name="status" required>
                            <option value="planned" {{ $allocation->status === 'planned' ? 'selected' : '' }}>{{ __('Planned') }}</option>
                            <option value="active" {{ $allocation->status === 'active' ? 'selected' : '' }}>{{ __('Active') }}</option>
                            <option value="completed" {{ $allocation->status === 'completed' ? 'selected' : '' }}>{{ __('Completed') }}</option>
                            <option value="cancelled" {{ $allocation->status === 'cancelled' ? 'selected' : '' }}>{{ __('Cancelled') }}</option>
                        </select>
                        <div class="invalid-feedback"></div>
                    </div>

                    <div class="mb-3">
                        <label for="notes" class="form-label">{{ __('Notes') }}</label>
                        <textarea class="form-control" id="notes" name="notes" rows="3">{{ $allocation->notes }}</textarea>
                        <div class="invalid-feedback"></div>
                    </div>

                    <div class="row mb-3">
                        <div class="col-md-6">
                            <div class="form-check form-switch">
                                <input class="form-check-input" type="checkbox" id="is_billable" name="is_billable"
                                       {{ $allocation->is_billable ? 'checked' : '' }}>
                                <label class="form-check-label" for="is_billable">
                                    {{ __('Billable') }}
                                </label>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="form-check form-switch">
                                <input class="form-check-input" type="checkbox" id="is_confirmed" name="is_confirmed"
                                       {{ $allocation->is_confirmed ? 'checked' : '' }}>
                                <label class="form-check-label" for="is_confirmed">
                                    {{ __('Confirmed Allocation') }}
                                </label>
                            </div>
                        </div>
                    </div>

                    <div class="d-flex gap-2">
                        <button type="submit" class="btn btn-primary">{{ __('Update Allocation') }}</button>
                        <a href="{{ route('pmcore.resources.index') }}" class="btn btn-label-secondary">{{ __('Cancel') }}</a>
                    </div>
                </form>
            </div>
        </div>

        <!-- Capacity Conflicts Alert -->
        @if($allocation->checkCapacityConflicts())
            <div class="card mt-3">
                <div class="card-body">
                    <div class="alert alert-warning mb-0">
                        <h6 class="alert-heading">{{ __('Capacity Conflicts Detected') }}</h6>
                        <p class="mb-0">{{ __('This resource is overallocated in some periods. Please review and adjust allocations.') }}</p>
                    </div>
                </div>
            </div>
        @endif
    </div>

    <!-- Allocation Info -->
    <div class="col-md-4 col-lg-6">
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">{{ __('Allocation Information') }}</h6>
            </div>
            <div class="card-body">
                <dl class="row mb-0">
                    <dt class="col-sm-5">{{ __('Created By') }}:</dt>
                    <dd class="col-sm-7">
                        @if($allocation->createdBy)
                            {{ $allocation->createdBy->name }}
                        @else
                            {{ __('System') }}
                        @endif
                    </dd>

                    <dt class="col-sm-5">{{ __('Created At') }}:</dt>
                    <dd class="col-sm-7">{{ $allocation->created_at->format('M d, Y H:i') }}</dd>

                    <dt class="col-sm-5">{{ __('Last Updated By') }}:</dt>
                    <dd class="col-sm-7">
                        @if($allocation->updatedBy)
                            {{ $allocation->updatedBy->name }}
                        @else
                            {{ __('System') }}
                        @endif
                    </dd>

                    <dt class="col-sm-5">{{ __('Last Updated At') }}:</dt>
                    <dd class="col-sm-7">{{ $allocation->updated_at->format('M d, Y H:i') }}</dd>
                </dl>
            </div>
        </div>
    </div>
</div>

@endsection
