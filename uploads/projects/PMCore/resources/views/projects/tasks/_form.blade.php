<!-- Task Form Offcanvas Component -->
@php
    $formId = $formId ?? 'addTaskForm';
    $containerId = $containerId ?? 'offcanvasAddTask';
    $labelId = $labelId ?? 'offcanvasAddTaskLabel';
@endphp

<!-- Add/Edit Task Offcanvas -->
<div class="offcanvas offcanvas-end" tabindex="-1" id="{{ $containerId }}" aria-labelledby="{{ $labelId }}">
    <div class="offcanvas-header">
        <h5 id="{{ $labelId }}" class="offcanvas-title">{{ __('Add Task') }}</h5>
        <button type="button" class="btn-close text-reset" data-bs-dismiss="offcanvas" aria-label="Close"></button>
    </div>
    <div class="offcanvas-body mx-0 flex-grow-0 pt-0 h-100">
        <form id="{{ $formId }}" class="pt-0">
            <div class="mb-3">
                <label class="form-label" for="task-title">{{ __('Task Title') }} <span class="text-danger">*</span></label>
                <input type="text" id="task-title" class="form-control" name="title" required />
            </div>
            
            <div class="mb-3">
                <label class="form-label" for="task-description">{{ __('Description') }}</label>
                <textarea id="task-description" class="form-control" name="description" rows="3"></textarea>
            </div>
            
            <div class="mb-3">
                <label class="form-label" for="task-status">{{ __('Status') }} <span class="text-danger">*</span></label>
                <select id="task-status" class="form-select" name="task_status_id" required>
                    <option value="">{{ __('Select Status') }}</option>
                    @foreach($statuses as $status)
                        <option value="{{ $status->id }}">{{ $status->name }}</option>
                    @endforeach
                </select>
            </div>
            
            <div class="mb-3">
                <label class="form-label" for="task-priority">{{ __('Priority') }} <span class="text-danger">*</span></label>
                <select id="task-priority" class="form-select" name="task_priority_id" required>
                    <option value="">{{ __('Select Priority') }}</option>
                    @foreach($priorities as $priority)
                        <option value="{{ $priority->id }}">{{ $priority->name }}</option>
                    @endforeach
                </select>
            </div>
            
            <div class="mb-3">
                <label class="form-label" for="task-assigned-to">{{ __('Assigned To') }}</label>
                <select id="task-assigned-to" class="select2 form-select" name="assigned_to_user_id">
                    <option value="">{{ __('Unassigned') }}</option>
                    @foreach($users as $user)
                        <option value="{{ $user['id'] }}">{{ $user['name'] }}</option>
                    @endforeach
                </select>
            </div>
            
            <div class="mb-3">
                <label class="form-label" for="task-due-date">{{ __('Due Date') }}</label>
                <input type="text" id="task-due-date" class="form-control flatpickr-input" name="due_date" placeholder="{{ __('Select date') }}" readonly="readonly" />
            </div>
            
            <div class="mb-3">
                <label class="form-label" for="task-estimated-hours">{{ __('Estimated Hours') }}</label>
                <input type="number" id="task-estimated-hours" class="form-control" name="estimated_hours" step="0.5" min="0" />
            </div>
            
            <div class="mb-3">
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="task-is-milestone" name="is_milestone" value="1">
                    <label class="form-check-label" for="task-is-milestone">
                        {{ __('Is Milestone') }}
                    </label>
                </div>
            </div>
            
            <button type="submit" class="btn btn-primary me-sm-3 me-1 data-submit">{{ __('Save') }}</button>
            <button type="reset" class="btn btn-outline-secondary" data-bs-dismiss="offcanvas">{{ __('Cancel') }}</button>
        </form>
    </div>
</div>
