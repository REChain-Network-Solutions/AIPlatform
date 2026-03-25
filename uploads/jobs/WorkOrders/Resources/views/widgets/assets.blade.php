<div class="card mb-3">
  <div class="card-header d-flex justify-content-between align-items-center">
    <span>Assets / Equipment</span>
    <form method="post" action="{{ route('workorders.assets.store', $work_order_id) }}" class="d-flex gap-2">
      @csrf
      <input class="form-control form-control-sm" name="asset_id" type="number" placeholder="Asset ID" required>
      <button class="btn btn-sm btn-primary">Link</button>
    </form>
  </div>
  <div class="card-body p-0">
    <table class="table table-sm mb-0">
      <thead><tr><th>Asset ID</th><th></th></tr></thead>
      <tbody>
      @forelse($rows as $r)
        <tr>
          <td>#{{ $r->asset_id }}</td>
          <td>
            <form method="post" action="{{ route('workorders.assets.destroy', [$work_order_id, $r->id]) }}">
              @csrf @method('DELETE')
              <button class="btn btn-sm btn-outline-danger">Unlink</button>
            </form>
          </td>
        </tr>
      @empty
        <tr><td colspan="2" class="text-muted">No assets linked.</td></tr>
      @endforelse
      </tbody>
    </table>
  </div>
</div>