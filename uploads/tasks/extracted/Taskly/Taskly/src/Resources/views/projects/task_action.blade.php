@permission('task show')
    <div class="action-btn me-2">
        <a data-size="lg" data-url="{{ route('tasks.show', [$task->project_id, $task->id]) }}" data-bs-toggle="tooltip"
            title="{{ __('View') }}" data-ajax-popup="true" data-title="{{ __('View Task Board') }}"
            class="mx-3 btn btn-sm align-items-center text-white bg-warning ">
            <i class="ti ti-eye"></i>
        </a>
    </div>
@endpermission
@permission('task edit')
    <div class="action-btn me-2">
        <a data-ajax-popup="true" data-size="lg" data-url="{{ route('tasks.edit', [$task->project_id, $task->id]) }}"
            class="btn btn-sm align-items-center text-white bg-info" data-bs-toggle="tooltip"
            data-title="{{ __('Edit Task Board') }}" title="{{ __('Edit') }}"><i class="ti ti-pencil"></i></a>

    </div>
@endpermission
@permission('task delete')
    <div class="action-btn">
        {!! Form::open(['method' => 'DELETE', 'route' => ['tasks.destroy', $task->project_id, $task->id]]) !!}
        <a href="#!" class="btn btn-sm   align-items-center text-white show_confirm bg-danger" data-bs-toggle="tooltip"
            title='Delete'>
            <i class="ti ti-trash"></i>
        </a>
        {!! Form::close() !!}
    </div>
@endpermission
