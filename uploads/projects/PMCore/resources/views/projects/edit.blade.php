@extends('layouts.layoutMaster')

@section('title', __('Edit Project'))

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
        project: @json($project),
        urls: {
            userSearch: @json(route('users.selectSearch')),
            clientSearch: @json(route('companies.selectSearch'))
        },
        labels: {
            success: @json(__('Success')),
            error: @json(__('Error')),
            updateSuccess: @json(__('Project updated successfully!')),
            validationError: @json(__('Please correct the errors below'))
        }
    };
</script>
@vite(['Modules/PMCore/resources/assets/js/project-form.js'])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Edit Project')"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Projects'), 'url' => route('pmcore.projects.index')],
        ['name' => $project->name, 'url' => route('pmcore.projects.show', $project)],
        ['name' => __('Edit'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<div class="row">
    <div class="col-md-12">
        <form id="projectForm" action="{{ route('pmcore.projects.update', $project) }}" method="POST">
            @csrf
            @method('PUT')

            @include('pmcore::projects._partials._form')

            <!-- Form Actions -->
            <div class="card">
                <div class="card-body">
                    <div class="d-flex gap-2">
                        <button type="submit" class="btn btn-primary">
                            <i class="bx bx-save me-1"></i>{{ __('Update Project') }}
                        </button>
                        <a href="{{ route('pmcore.projects.show', $project) }}" class="btn btn-label-secondary">
                            <i class="bx bx-x me-1"></i>{{ __('Cancel') }}
                        </a>
                    </div>
                </div>
            </div>
        </form>
    </div>
</div>
@endsection
