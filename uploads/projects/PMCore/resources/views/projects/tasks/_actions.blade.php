<div class="dropdown">
  <button type="button" class="btn btn-sm btn-outline-secondary dropdown-toggle" data-bs-toggle="dropdown">
    {{ __('Actions') }}
  </button>
  <ul class="dropdown-menu">
    <li>
      <a class="dropdown-item edit-task" href="javascript:void(0)" data-id="{{ $task->id }}">
        <i class="bx bx-edit-alt me-1"></i>
        {{ __('Edit') }}
      </a>
    </li>
    
    @if(!$task->completed_at)
      {{-- Show Complete button only if task is not completed --}}
      <li>
        <a class="dropdown-item complete-task" href="javascript:void(0)" data-id="{{ $task->id }}">
          <i class="bx bx-check me-1"></i>
          {{ __('Complete') }}
        </a>
      </li>
      
      {{-- Show Start/Stop time only if task is not completed --}}
      @if($task->time_started_at)
        <li>
          <a class="dropdown-item stop-time" href="javascript:void(0)" data-id="{{ $task->id }}">
            <i class="bx bx-stop me-1"></i>
            {{ __('Stop Time') }}
          </a>
        </li>
      @else
        <li>
          <a class="dropdown-item start-time" href="javascript:void(0)" data-id="{{ $task->id }}">
            <i class="bx bx-play me-1"></i>
            {{ __('Start Time') }}
          </a>
        </li>
      @endif
    @else
      {{-- Show status for completed tasks --}}
      <li>
        <span class="dropdown-item-text text-success">
          <i class="bx bx-check-circle me-1"></i>
          {{ __('Completed') }}
        </span>
      </li>
      @if($task->completed_at)
        <li>
          <span class="dropdown-item-text text-muted small">
            {{ __('Completed on') }}: {{ $task->completed_at->format('M d, Y H:i') }}
          </span>
        </li>
      @endif
    @endif
    
    <li><hr class="dropdown-divider"></li>
    <li>
      <a class="dropdown-item text-danger delete-task" href="javascript:void(0)" data-id="{{ $task->id }}">
        <i class="bx bx-trash me-1"></i>
        {{ __('Delete') }}
      </a>
    </li>
  </ul>
</div>
