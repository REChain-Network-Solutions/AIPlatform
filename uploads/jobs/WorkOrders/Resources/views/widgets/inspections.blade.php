<div class="card mb-3">
  <div class="card-header d-flex justify-content-between align-items-center">
    <span>Inspections / QA</span>
    <form method="post" action="{{ route('workorders.inspections.store', $work_order_id) }}" class="d-flex gap-2">
      @csrf
      <input class="form-control form-control-sm" name="inspection_id" placeholder="Inspection ID">
      <input class="form-control form-control-sm" name="template_name" placeholder="Template">
      <input class="form-control form-control-sm" name="completed_at" type="datetime-local">
      <input class="form-control form-control-sm" name="pdf_path" placeholder="PDF path">
      <button class="btn btn-sm btn-primary">Record</button>
    </form>
  </div>
  <div class="card-body p-0">
    <table class="table table-sm mb-0">
      <thead><tr><th>Template</th><th>Completed</th><th>PDF</th></tr></thead>
      <tbody>
      @forelse($rows as $r)
        <tr>
          <td>{{ $r->template_name ?? ('#'.$r->inspection_id) }}</td>
          <td>{{ $r->completed_at }}</td>
          <td>@if($r->pdf_path)<a href="{{ asset($r->pdf_path) }}" target="_blank">Open</a>@endif</td>
        </tr>
      @empty
        <tr><td colspan="3" class="text-muted">No inspections recorded.</td></tr>
      @endforelse
      </tbody>
    </table>
  </div>
</div>