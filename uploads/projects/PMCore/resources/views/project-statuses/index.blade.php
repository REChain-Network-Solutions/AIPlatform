@extends('layouts.layoutMaster')

@section('title', __('Project Statuses'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-responsive-bs5/responsive.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-buttons-bs5/buttons.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/sweetalert2/sweetalert2.scss'])
    <style>
        .sortable-ghost {
            opacity: 0.4;
            background: #f8f9fa;
        }
        .datatables-project-statuses tbody tr:hover {
            background-color: #f8f9fa;
        }
        .drag-handle {
            color: #697a8d;
            transition: color 0.2s;
            cursor: move;
        }
        .drag-handle:hover {
            color: #566a7f;
        }
        .datatables-project-statuses tbody tr.dragging {
            opacity: 0.8;
        }
    </style>
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables-bootstrap5.js'])
    @vite(['resources/assets/vendor/libs/sweetalert2/sweetalert2.js'])
    @vite(['resources/assets/vendor/libs/sortablejs/sortable.js'])
@endsection

@section('page-script')
    <script>
        window.pageData = {
            urls: {
                datatableUrl: @json(route('pmcore.project-statuses.getDataAjax')),
                storeUrl: @json(route('pmcore.project-statuses.store')),
                updateUrl: @json(route('pmcore.project-statuses.update', ['projectStatus' => '__ID__'])),
                deleteUrl: @json(route('pmcore.project-statuses.destroy', ['projectStatus' => '__ID__'])),
                toggleActiveUrl: @json(route('pmcore.project-statuses.toggle-active', ['projectStatus' => '__ID__'])),
                sortOrderUrl: @json(route('pmcore.project-statuses.update-sort-order')),
                setDefaultUrl: @json(route('pmcore.project-statuses.set-default'))
            },
            labels: {
                confirmDelete: @json(__('Are you sure you want to delete this project status?')),
                confirmToggle: @json(__('Are you sure you want to change the status?')),
                deleteSuccess: @json(__('Project status deleted successfully!')),
                toggleSuccess: @json(__('Status updated successfully!')),
                sortSuccess: @json(__('Sort order updated successfully!')),
                error: @json(__('An error occurred. Please try again.')),
                validationError: @json(__('Please correct the errors below')),
                createSuccess: @json(__('Project status created successfully!')),
                updateSuccess: @json(__('Project status updated successfully!')),
                yes: @json(__('Yes')),
                no: @json(__('No')),
                searchPlaceholder: @json(__('Search Project Statuses')),
                addTitle: @json(__('Add Project Status')),
                editTitle: @json(__('Edit Project Status')),
                success: @json(__('Success!')),
                ok: @json(__('OK')),
                areYouSure: @json(__('Are you sure?')),
                yesDelete: @json(__('Yes, delete it!')),
                yesChange: @json(__('Yes, change it!'))
            }
        };
    </script>
    @vite(['Modules/PMCore/resources/assets/js/project-statuses.js'])
@endsection

@section('content')
@php
  $breadcrumbs = [
    ['name' => __('Master Data'), 'url' => route('master-data.index')],
    ['name' => __('Project Management'), 'url' => '#']
  ];
@endphp

<x-breadcrumb
  :title="__('Project Statuses')"
  :breadcrumbs="$breadcrumbs"
  :homeUrl="route('dashboard')"
/>

<div class="card">
    <div class="card-header">
        <div class="d-flex justify-content-between align-items-center">
            <h5 class="card-title mb-0">{{ __('All Project Statuses') }}</h5>
            <button class="btn btn-primary" data-bs-toggle="offcanvas" data-bs-target="#projectStatusFormOffcanvas">
                <i class="bx bx-plus"></i> {{ __('Add New Status') }}
            </button>
        </div>
    </div>
    <div class="card-body">
        <div class="alert alert-info">
            <i class="bx bx-info-circle me-2"></i>
            {{ __('You can drag and drop rows to reorder project statuses. The order will be saved automatically.') }}
        </div>

        <div class="card-datatable table-responsive">
            <table class="table table-hover border-top datatables-project-statuses">
                <thead>
                    <tr>
                        <th>{{ __('Order') }}</th>
                        <th>{{ __('Status') }}</th>
                        <th>{{ __('Description') }}</th>
                        <th>{{ __('Projects') }}</th>
                        <th>{{ __('Active') }}</th>
                        <th>{{ __('Default') }}</th>
                        <th>{{ __('Completed') }}</th>
                        <th>{{ __('Actions') }}</th>
                    </tr>
                </thead>
            </table>
        </div>
    </div>
</div>

<div class="offcanvas offcanvas-end" tabindex="-1" id="projectStatusFormOffcanvas">
    <div class="offcanvas-header">
        <h5 class="offcanvas-title" id="offcanvasTitle">{{ __('Add Project Status') }}</h5>
        <button type="button" class="btn-close text-reset" data-bs-dismiss="offcanvas"></button>
    </div>
    <div class="offcanvas-body">
        <form id="projectStatusForm">
            <input type="hidden" id="statusId" name="id">

            <div class="mb-3">
                <label for="statusName" class="form-label">{{ __('Status Name') }} <span class="text-danger">*</span></label>
                <input type="text" class="form-control" id="statusName" name="name" required>
                <div class="invalid-feedback"></div>
            </div>

            <div class="mb-3">
                <label for="statusDescription" class="form-label">{{ __('Description') }}</label>
                <textarea class="form-control" id="statusDescription" name="description" rows="3"></textarea>
                <div class="invalid-feedback"></div>
            </div>

            <div class="mb-3">
                <label for="statusColor" class="form-label">{{ __('Color') }} <span class="text-danger">*</span></label>
                <input type="color" class="form-control" id="statusColor" name="color" value="#007bff" required>
                <div class="invalid-feedback"></div>
            </div>

            <div class="mb-3">
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="statusActive" name="is_active" checked>
                    <label class="form-check-label" for="statusActive">
                        {{ __('Active') }}
                    </label>
                </div>
            </div>

            <div class="mb-3">
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="statusDefault" name="is_default">
                    <label class="form-check-label" for="statusDefault">
                        {{ __('Default Status') }}
                    </label>
                </div>
                <small class="text-muted">{{ __('Only one status can be set as default') }}</small>
            </div>

            <div class="mb-3">
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="statusCompleted" name="is_completed">
                    <label class="form-check-label" for="statusCompleted">
                        {{ __('Represents Completion') }}
                    </label>
                </div>
                <small class="text-muted">{{ __('Mark if this status indicates project completion') }}</small>
            </div>

            <div class="d-flex gap-2">
                <button type="submit" class="btn btn-primary flex-fill">
                    <i class="bx bx-save me-1"></i>{{ __('Save') }}
                </button>
                <button type="button" class="btn btn-label-secondary flex-fill" data-bs-dismiss="offcanvas">
                    {{ __('Cancel') }}
                </button>
            </div>
        </form>
    </div>
</div>

@endsection
