@extends('client.layouts.app')
@section('content')
<div class="container" style="max-width:980px">
  <h3>@term('work_order')s</h3>
  <table class="table table-sm align-middle">
    <thead><tr><th>ID</th><th>Status</th><th>Scheduled</th><th></th></tr></thead>
    <tbody>
      @forelse($rows as $r)
        <tr>
          <td>#{{ $r->id }}</td>
          <td>{{ $r->status }}</td>
          <td>{{ $r->scheduled_at }}</td>
          <td><a class="btn btn-sm btn-outline-primary" href="{{ route('client.workorders.show',$r->id) }}">Open</a></td>
        </tr>
      @empty
        <tr><td colspan="4" class="text-muted">No @term('work_order')s yet.</td></tr>
      @endforelse
    </tbody>
  </table>
</div>
@endsection