<div class="card mb-3">
  <div class="card-header d-flex justify-content-between align-items-center">
    <span>Assigned Contractors</span>
    <a class="btn btn-sm btn-primary" href="{{ route('workorders.assign.redirect', $work_order_id) }}">Assign contractor</a>
  </div>
  <div class="card-body p-0">
    <div class="table-responsive">
      <table class="table table-sm mb-0">
        <thead><tr><th>Contractor</th><th>Role</th><th>Scheduled</th><th>Status</th></tr></thead>
        <tbody>
          @forelse($rows as $a)
            <tr>
              <td>{{ optional($a->contractor)->name ?? ('#'.$a->contractor_id) }}</td>
              <td>{{ $a->role }}</td>
              <td>{{ $a->scheduled_at }}</td>
              <td>{{ $a->status }}</td>
            </tr>
          @empty
            <tr><td colspan="4" class="text-muted">No assignments yet.</td></tr>
          @endforelse
        </tbody>
      </table>
    </div>
  </div>
</div>
