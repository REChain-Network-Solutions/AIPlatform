<div class="card mb-3">
  <div class="card-header">Recurrence</div>
  <div class="card-body">
    <form method="post" action="{{ route('workorders.recurrence.store', $work_order_id) }}" class="row g-2">
      @csrf
      <div class="col-md-8">
        <input class="form-control" name="rrule" value="{{ $rec->rrule ?? 'FREQ=WEEKLY;INTERVAL=1;BYHOUR=9;BYMINUTE=0' }}">
      </div>
      <div class="col-md-2 form-check">
        <input class="form-check-input" type="checkbox" id="is_active" name="is_active" {{ ($rec && $rec->is_active)?'checked':'' }}>
        <label class="form-check-label" for="is_active">Active</label>
      </div>
      <div class="col-md-2">
        <button class="btn btn-primary w-100">Save</button>
      </div>
    </form>
    @if($rec)
      <div class="text-muted mt-2">Next run: {{ $rec->next_run_at }} | Last run: {{ $rec->last_run_at }}</div>
    @endif
  </div>
</div>
