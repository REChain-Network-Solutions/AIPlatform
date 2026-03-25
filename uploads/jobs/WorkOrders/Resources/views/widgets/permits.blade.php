<div class="card mb-3">
  <div class="card-header d-flex justify-content-between align-items-center">
    <span>Permits</span>
    <form method="post" action="{{ route('workorders.permits.store', $work_order_id) }}" class="d-flex gap-2">
      @csrf
      <input class="form-control form-control-sm" name="type" placeholder="Type">
      <select class="form-select form-select-sm" name="status">
        <option value="pending">Pending</option>
        <option value="approved">Approved</option>
        <option value="rejected">Rejected</option>
        <option value="expired">Expired</option>
      </select>
      <input class="form-control form-control-sm" name="permit_number" placeholder="Permit #">
      <input class="form-control form-control-sm" name="valid_from" type="date">
      <input class="form-control form-control-sm" name="valid_to" type="date">
      <button class="btn btn-sm btn-primary">Save</button>
    </form>
  </div>
  <div class="card-body p-0">
    <table class="table table-sm mb-0">
      <thead><tr><th>Type</th><th>Status</th><th>Permit #</th><th>Valid</th></tr></thead>
      <tbody>
      @forelse($rows as $r)
        <tr>
          <td>{{ $r->type }}</td>
          <td>{{ ucfirst($r->status) }}</td>
          <td>{{ $r->permit_number }}</td>
          <td>{{ $r->valid_from }} → {{ $r->valid_to }}</td>
        </tr>
      @empty
        <tr><td colspan="4" class="text-muted">No permits recorded.</td></tr>
      @endforelse
      </tbody>
    </table>
  </div>
</div>