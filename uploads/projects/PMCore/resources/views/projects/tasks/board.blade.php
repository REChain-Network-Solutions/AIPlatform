@extends('layouts.layoutMaster')

@section('title', __('Task Board'))

@section('vendor-style')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/datatables-responsive-bs5/responsive.bootstrap5.scss'])
    @vite(['resources/assets/vendor/libs/select2/select2.scss'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.scss'])
    @vite(['resources/assets/vendor/libs/sweetalert2/sweetalert2.scss'])
    @vite(['resources/assets/vendor/libs/sortablejs/sortable.scss'])
@endsection

@section('vendor-script')
    @vite(['resources/assets/vendor/libs/datatables-bs5/datatables-bootstrap5.js'])
    @vite(['resources/assets/vendor/libs/select2/select2.js'])
    @vite(['resources/assets/vendor/libs/flatpickr/flatpickr.js'])
    @vite(['resources/assets/vendor/libs/sweetalert2/sweetalert2.js'])
    @vite(['resources/assets/vendor/libs/sortablejs/sortable.js'])
@endsection

@section('page-script')
    @vite(['Modules/PMCore/resources/assets/js/task-form-handler.js'])
    @vite(['Modules/PMCore/resources/assets/js/project-tasks-board.js'])
@endsection

@section('page-style')
    <style>
        .kanban-board {
            overflow-x: auto;
            min-height: 70vh;
        }

        .kanban-column {
            min-width: 300px;
            max-width: 350px;
            background: var(--bs-gray-100);
            border: 1px solid var(--bs-border-color);
            border-radius: 8px;
            padding: 1rem;
            margin-right: 1rem;
            flex-shrink: 0;
        }

        [data-bs-theme="dark"] .kanban-column {
            background: var(--bs-dark);
            border-color: var(--bs-border-color);
        }

        .kanban-header {
            border-bottom: 2px solid var(--bs-border-color);
            padding-bottom: 0.5rem;
        }

        .kanban-tasks {
            min-height: 400px;
            padding-top: 1rem;
        }

        .kanban-task {
            cursor: move;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            border: 1px solid var(--bs-border-color);
        }

        .kanban-task:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-color: var(--bs-primary);
        }

        [data-bs-theme="dark"] .kanban-task:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            border-color: var(--bs-primary);
        }

        .sortable-chosen {
            opacity: 0.5;
        }

        .sortable-ghost {
            opacity: 0.3;
            background: var(--bs-secondary-bg);
            border: 2px dashed var(--bs-secondary);
        }

        .kanban-task .dropdown-toggle::after {
            display: none;
        }

        @media (max-width: 768px) {
            .kanban-board {
                flex-direction: column;
            }

            .kanban-column {
                min-width: 100%;
                margin-right: 0;
                margin-bottom: 1rem;
            }
        }
    </style>
@endsection

@section('content')
<x-breadcrumb
    :title="__('Task Board') . ' - ' . $project->name"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Projects'), 'url' => route('pmcore.projects.index')],
        ['name' => $project->name, 'url' => route('pmcore.projects.show', $project->id)],
        ['name' => __('Task Board'), 'url' => '']
    ]"
    :home-url="route('dashboard')"
>
    <x-slot name="actions">
        <a href="{{ route('pmcore.projects.tasks.index', $project->id) }}" class="btn btn-outline-primary">
            <i class="bx bx-list-ul me-1"></i>{{ __('View Tasks') }}
        </a>
        <button type="button" class="btn btn-primary" data-bs-toggle="offcanvas" data-bs-target="#offcanvasTaskForm" id="add-task-btn">
            <i class="bx bx-plus me-1"></i>{{ __('Add Task') }}
        </button>
    </x-slot>
</x-breadcrumb>

<!-- Kanban Board -->
<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-body p-0">
                <div class="kanban-board d-flex gap-3 p-3" id="kanban-board">
                    <!-- Kanban columns will be populated by JavaScript -->
                </div>
            </div>
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

<script>
// Define pageData immediately when this script executes
window.pageData = {
    projectId: {{ $project->id }},
    statuses: @json($statuses),
    priorities: @json($priorities),
    users: @json($users),
    urls: {
        getDataAjax: @json(route('pmcore.projects.tasks.getDataAjax', $project->id)),
        store: @json(route('pmcore.projects.tasks.store', $project->id)),
        update: @json(route('pmcore.projects.tasks.update', [$project->id, '__ID__'])),
        destroy: @json(route('pmcore.projects.tasks.destroy', [$project->id, '__ID__'])),
        show: @json(route('pmcore.projects.tasks.show', [$project->id, '__ID__'])),
        complete: @json(route('pmcore.projects.tasks.complete', [$project->id, '__ID__'])),
        start: @json(route('pmcore.projects.tasks.start', [$project->id, '__ID__'])),
        stop: @json(route('pmcore.projects.tasks.stop', [$project->id, '__ID__'])),
        reorder: @json(route('pmcore.projects.tasks.reorder', $project->id)),
        // URLs for TaskFormHandler
        tasksStore: @json(route('pmcore.projects.tasks.store', $project->id)),
        tasksUpdate: @json(route('pmcore.projects.tasks.update', [$project->id, '__ID__'])),
        tasksShow: @json(route('pmcore.projects.tasks.show', [$project->id, '__ID__']))
    },
    labels: {
        addTask: @json(__('Add Task')),
        editTask: @json(__('Edit Task')),
        loading: @json(__('Loading...')),
        noTasks: @json(__('No tasks found')),
        dragToReorder: @json(__('Drag tasks to reorder')),
        confirmDelete: @json(__('Are you sure you want to delete this task?')),
        confirmComplete: @json(__('Are you sure you want to complete this task?')),
        confirmStart: @json(__('Are you sure you want to start this task?')),
        confirmStop: @json(__('Are you sure you want to stop this task?')),
        taskCreated: @json(__('Task created successfully!')),
        taskUpdated: @json(__('Task updated successfully!')),
        taskDeleted: @json(__('Task deleted successfully!')),
        taskCompleted: @json(__('Task completed successfully!')),
        taskStarted: @json(__('Task started successfully!')),
        taskStopped: @json(__('Task stopped successfully!')),
        tasksReordered: @json(__('Tasks reordered successfully!')),
        error: @json(__('An error occurred. Please try again.')),
        unassigned: @json(__('Unassigned')),
        assignedTo: @json(__('Assigned to')),
        dueDate: @json(__('Due Date')),
        estimatedHours: @json(__('Estimated Hours')),
        milestone: @json(__('Milestone')),
        overdue: @json(__('Overdue')),
        completed: @json(__('Completed')),
        running: @json(__('Running'))
    }
};

// Dispatch a custom event to notify that pageData is ready
if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('pageDataReady', { detail: window.pageData }));
}
</script>

@endsection
