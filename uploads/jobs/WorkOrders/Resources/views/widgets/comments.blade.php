
<div class="card mb-3">
  <div class="card-header">@term('work_order') Activity</div>
  <div class="card-body p-0">
    <table class="table table-sm mb-0"><tbody>
      @forelse($rows as $r)
        <tr><td>
          <div class="small text-muted">{{ $r->created_at }} — User #{{ $r->user_id }}</div>
          <div>{{ $r->body }}</div>
          @if($r->attachment_path)<div><a href="{{ asset($r->attachment_path) }}" target="_blank">Attachment</a></div>@endif
        </td></tr>
      @empty
        <tr><td class="text-muted">No activity yet.</td></tr>
      @endforelse
    </tbody></table>
  </div>
  <div class="card-footer">
    <form method="post" action="{{ route('workorders.comments.store', $id) }}" enctype="multipart/form-data" class="d-flex gap-2">
      @csrf
      <input class="form-control" name="body" placeholder="Add a note...">
      <input class="form-control" type="file" name="attachment">
      <button class="btn btn-primary">Post</button>
    </form>
  </div>
</div>
