@extends('layouts.layoutMaster')

@section('title', __('Projects'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-responsive-bs5/responsive.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-buttons-bs5/buttons.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/select2/select2.scss'])
    @vite(['resources/assets/vendor/libs/sweetalert2/sweetalert2.scss'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables-bootstrap5.js'])
    @vite(['resources/assets/vendor/libs/select2/select2.js'])
    @vite(['resources/assets/vendor/libs/sweetalert2/sweetalert2.js'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.js'])
@endsection

@section('page-script')
    <script>
        window.pageData = {
            urls: {
                projectData: @json(route('pmcore.projects.data')),
                projectStore: @json(route('pmcore.projects.store')),
                projectUpdate: @json(route('pmcore.projects.update', ['project' => '__ID__'])),
                projectDelete: @json(route('pmcore.projects.destroy', ['project' => '__ID__'])),
                projectShow: @json(route('pmcore.projects.show', ['project' => '__ID__'])),
                projectEdit: @json(route('pmcore.projects.edit', ['project' => '__ID__']))
            },
            labels: {
                confirmDelete: @json(__('Are you sure you want to delete this project?')),
                deleteSuccess: @json(__('Project deleted successfully!')),
                error: @json(__('An error occurred. Please try again.')),
                search: @json(__('Search')),
                searchPlaceholder: @json(__('Search Projects...')),
                lengthMenu: @json(__('Show _MENU_ entries')),
                info: @json(__('Showing _START_ to _END_ of _TOTAL_ entries')),
                infoEmpty: @json(__('Showing 0 to 0 of 0 entries')),
                infoFiltered: @json(__('(filtered from _MAX_ total entries)')),
                loadingRecords: @json(__('Loading...')),
                processing: @json(__('Processing...')),
                zeroRecords: @json(__('No matching records found')),
                emptyTable: @json(__('No data available in table')),
                paginate: {
                    first: @json(__('First')),
                    last: @json(__('Last')),
                    next: @json(__('Next')),
                    previous: @json(__('Previous'))
                },
                yesDeleteIt: @json(__('Yes, delete it!')),
                cancel: @json(__('Cancel')),
                deleted: @json(__('Deleted!'))
            }
        };
    </script>
    @vite(['Modules/PMCore/resources/assets/js/project-list.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Projects')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Projects'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

@include('pmcore::projects._partials._stats', ['stats' => $stats])

<!-- Projects Table -->
<div class="card">
    <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="mb-0">{{ __('Manage Projects') }}</h5>
        @can('pmcore.create-project')
        <a href="{{ route('pmcore.projects.create') }}" class="btn btn-primary">
            <i class="bx bx-plus me-1"></i>{{ __('Add New Project') }}
        </a>
        @endcan
    </div>
    <div class="card-datatable table-responsive">
        <table class="table table-hover border-top datatables-projects">
            <thead>
                <tr>
                    <th>{{ __('ID') }}</th>
                    <th>{{ __('Name') }}</th>
                    <th>{{ __('Code') }}</th>
                    <th>{{ __('Client') }}</th>
                    <th>{{ __('Status') }}</th>
                    <th>{{ __('Priority') }}</th>
                    <th>{{ __('Start Date') }}</th>
                    <th>{{ __('End Date') }}</th>
                    <th>{{ __('Progress') }}</th>
                    <th>{{ __('Actions') }}</th>
                </tr>
            </thead>
        </table>
    </div>
</div>

@endsection
