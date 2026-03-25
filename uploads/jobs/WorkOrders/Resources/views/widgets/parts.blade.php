<div class="card mb-3">
  <div class="card-header d-flex justify-content-between align-items-center">
    <span>Parts & Materials</span>
    <form method="post" action="{{ route('workorders.parts.store', $work_order_id) }}" class="d-flex gap-2">
      @csrf
      <input class="form-control form-control-sm" name="item_id" placeholder="Item ID">
      <input class="form-control form-control-sm" name="item_name" placeholder="Item Name">
      <input class="form-control form-control-sm" name="qty" type="number" step="0.001" placeholder="Qty" required>
      <input class="form-control form-control-sm" name="unit_price" type="number" step="0.01" placeholder="Unit Price">
      <input class="form-control form-control-sm" name="source_location" placeholder="Location">
      <button class="btn btn-sm btn-primary">Add</button>
    </form>
  </div>
  <div class="card-body p-0">
    <table class="table table-sm mb-0">
      <thead><tr><th>Item</th><th>Qty</th><th>Unit Price</th><th>Source</th><th></th></tr></thead>
      <tbody>
      @forelse($rows as $r)
        <tr>
          <td>{{ $r->item_name ?? ('#'.$r->item_id) }}</td>
          <td>{{ $r->qty }}</td>
          <td>{{ $r->unit_price }}</td>
          <td>{{ $r->source_location }}</td>
          <td>
            <form method="post" action="{{ route('workorders.parts.destroy', [$work_order_id, $r->id]) }}">
              @csrf @method('DELETE')
              <button class="btn btn-sm btn-outline-danger">Remove</button>
            </form>
          </td>
        </tr>
      @empty
        <tr><td colspan="5" class="text-muted">No parts recorded.</td></tr>
      @endforelse
      </tbody>
    </table>
  </div>
</div>