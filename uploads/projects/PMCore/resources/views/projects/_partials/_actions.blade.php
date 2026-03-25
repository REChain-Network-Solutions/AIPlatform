@php
    $actions = [];
    
    // View action - always show if user can view the project
    if (auth()->user()->can('view', $project)) {
        $actions[] = [
            'label' => __('View'),
            'icon' => 'bx bx-show',
            'url' => route('pmcore.projects.show', $project->id)
        ];
    }
    
    // Edit action
    if (auth()->user()->can('update', $project)) {
        $actions[] = [
            'label' => __('Edit'),
            'icon' => 'bx bx-edit',
            'url' => route('pmcore.projects.edit', $project->id)
        ];
    }
    
    // Delete action
    if (auth()->user()->can('delete', $project)) {
        $actions[] = [
            'label' => __('Delete'),
            'icon' => 'bx bx-trash',
            'onclick' => 'deleteProject(' . $project->id . ')',
            'class' => 'text-danger'
        ];
    }
@endphp

<x-datatable-actions
    :id="$project->id"
    :actions="$actions"
/>