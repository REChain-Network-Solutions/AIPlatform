@extends('layouts.layoutMaster')

@section('title', __('Project Tasks'))

@section('vendor-style')
  @vite([
    'resources/assets/vendor/libs/datatables-bs5/datatables.bootstrap5.scss',
    'resources/assets/vendor/libs/datatables-responsive-bs5/responsive.bootstrap5.scss',
    'resources/assets/vendor/libs/select2/select2.scss',
    'resources/assets/vendor/libs/flatpickr/flatpickr.scss',
    'resources/assets/vendor/libs/sweetalert2/sweetalert2.scss'
  ])
@endsection

@section('content')
<x-breadcrumb
    :title="__('Project Tasks') . ' - ' . $project->name"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Projects'), 'url' => route('pmcore.projects.index')],
        ['name' => $project->name, 'url' => route('pmcore.projects.show', $project->id)],
        ['name' => __('Tasks'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<div class="row g-4 mb-4">
  <div class="col-sm-6 col-xl-3">
    <div class="card">
      <div class="card-body">
        <div class="d-flex justify-content-between">
          <div class="me-1">
            <p class="text-heading mb-2">{{ __('Total Tasks') }}</p>
            <div class="d-flex align-items-center">
              <h4 class="mb-2 me-1 display-6">{{ $project->tasks()->count() }}</h4>
            </div>
          </div>
          <div class="avatar">
            <div class="avatar-initial bg-label-primary rounded">
              <i class="bx bx-task fs-4"></i>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="col-sm-6 col-xl-3">
    <div class="card">
      <div class="card-body">
        <div class="d-flex justify-content-between">
          <div class="me-1">
            <p class="text-heading mb-2">{{ __('Completed Tasks') }}</p>
            <div class="d-flex align-items-center">
              <h4 class="mb-2 me-1 display-6">{{ $project->completedTasks()->count() }}</h4>
            </div>
          </div>
          <div class="avatar">
            <div class="avatar-initial bg-label-success rounded">
              <i class="bx bx-check-circle fs-4"></i>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="col-sm-6 col-xl-3">
    <div class="card">
      <div class="card-body">
        <div class="d-flex justify-content-between">
          <div class="me-1">
            <p class="text-heading mb-2">{{ __('Pending Tasks') }}</p>
            <div class="d-flex align-items-center">
              <h4 class="mb-2 me-1 display-6">{{ $project->pendingTasks()->count() }}</h4>
            </div>
          </div>
          <div class="avatar">
            <div class="avatar-initial bg-label-warning rounded">
              <i class="bx bx-time fs-4"></i>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="col-sm-6 col-xl-3">
    <div class="card">
      <div class="card-body">
        <div class="d-flex justify-content-between">
          <div class="me-1">
            <p class="text-heading mb-2">{{ __('Overdue Tasks') }}</p>
            <div class="d-flex align-items-center">
              <h4 class="mb-2 me-1 display-6">{{ $project->overdueTasks()->count() }}</h4>
            </div>
          </div>
          <div class="avatar">
            <div class="avatar-initial bg-label-danger rounded">
              <i class="bx bx-error fs-4"></i>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="card">
  <div class="card-header">
    <h5 class="card-title mb-0">{{ __('Tasks List') }}</h5>
  </div>
  <div class="card-body">
    <div class="row mb-3">
      <div class="col-md-6">
        <a href="{{ route('pmcore.projects.tasks.board', $project->id) }}" class="btn btn-outline-primary">
          <i class="bx bx-grid-alt me-2"></i>{{ __('Board View') }}
        </a>
      </div>
      <div class="col-md-6 text-end">
        <button class="btn btn-primary" data-bs-toggle="offcanvas" data-bs-target="#offcanvasTaskForm">
          <i class="bx bx-plus me-2"></i>{{ __('Add Task') }}
        </button>
      </div>
    </div>

    <div class="table-responsive">
      <table class="table table-striped datatables-tasks" id="tasksTable">
        <thead>
          <tr>
            <th>{{ __('Task') }}</th>
            <th>{{ __('Status') }}</th>
            <th>{{ __('Priority') }}</th>
            <th>{{ __('Assigned To') }}</th>
            <th>{{ __('Due Date') }}</th>
            <th>{{ __('Actions') }}</th>
          </tr>
        </thead>
      </table>
    </div>
  </div>
</div>

<!-- Include Task Form Component -->
@include('pmcore::projects.tasks._form', [
    'formId' => 'taskForm',
    'containerId' => 'offcanvasTaskForm',
    'labelId' => 'offcanvasTaskFormLabel',
    'statuses' => $statuses,
    'priorities' => $priorities,
    'users' => $users
])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables-bootstrap5.js'])
    @vite(['resources/assets/vendor/libs/select2/select2.js'])
    @vite(['resources/assets/vendor/libs/sweetalert2/sweetalert2.js'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.js'])
@endsection

@section('page-script')
<script>
// Define pageData immediately when this script executes
window.pageData = {
    projectId: {{ $project->id }},
    urls: {
        tasksData: @json(route('pmcore.projects.tasks.getDataAjax', $project->id)),
        tasksStore: @json(route('pmcore.projects.tasks.store', $project->id)),
        tasksUpdate: @json(route('pmcore.projects.tasks.update', [$project->id, '__ID__'])),
        tasksDestroy: @json(route('pmcore.projects.tasks.destroy', [$project->id, '__ID__'])),
        tasksComplete: @json(route('pmcore.projects.tasks.complete', [$project->id, '__ID__'])),
        tasksStart: @json(route('pmcore.projects.tasks.start', [$project->id, '__ID__'])),
        tasksStop: @json(route('pmcore.projects.tasks.stop', [$project->id, '__ID__'])),
        tasksShow: @json(route('pmcore.projects.tasks.show', [$project->id, '__ID__'])),
        boardView: @json(route('pmcore.projects.tasks.board', $project->id))
    },
    labels: {
        loading: @json(__('Loading...')),
        error: @json(__('An error occurred. Please try again.')),
        saveSuccess: @json(__('Task saved successfully!')),
        updateSuccess: @json(__('Task updated successfully!')),
        deleteSuccess: @json(__('Task deleted successfully!')),
        completeSuccess: @json(__('Task completed successfully!')),
        startSuccess: @json(__('Task started successfully!')),
        stopSuccess: @json(__('Task stopped successfully!')),
        confirmDelete: @json(__('Are you sure you want to delete this task?')),
        completeConfirm: @json(__('Are you sure you want to complete this task?')),
        startConfirm: @json(__('Are you sure you want to start this task?')),
        stopConfirm: @json(__('Are you sure you want to stop this task?')),
        editTask: @json(__('Edit Task')),
        addTask: @json(__('Add New Task'))
    },
    statuses: @json($statuses ?? []),
    priorities: @json($priorities ?? []),
    users: @json($users ?? [])
};

// Dispatch a custom event to notify that pageData is ready
if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('pageDataReady', { detail: window.pageData }));
}
</script>
    @vite(['Modules/PMCore/resources/assets/js/task-form-handler.js'])
    @vite(['Modules/PMCore/resources/assets/js/project-tasks.js'])
@endsection
