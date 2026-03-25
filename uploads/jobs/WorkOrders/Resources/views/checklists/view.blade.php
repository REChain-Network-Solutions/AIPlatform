@extends('layouts.app')
@section('content')
<div class="container" style="max-width:820px">
  <div class="d-flex justify-content-between align-items-center">
    <h4>@term('work_order') #{{ $woId }} — Checklist</h4>
    <a class="btn btn-outline-secondary btn-sm" href="{{ route('workorders.checklists.picker',$woId) }}">Choose template</a>
  </div>
  @if(!$wc)
    <div class="alert alert-info mt-3">No checklist attached yet.</div>
  @else
    <table class="table table-sm align-middle mt-3">
      <thead><tr><th style="width:60%">Item</th><th>Status</th><th>Notes</th><th></th></tr></thead>
      <tbody>
      @foreach($items as $i)
        <tr>
          <td>{{ $i->text }} @if($i->required)<span class="text-danger">*</span>@endif</td>
          <td>
            <form method="post" action="{{ route('workorders.checklists.items.update', [$woId,$i->id]) }}" class="d-flex gap-2">@csrf
              <select name="status" class="form-select form-select-sm" onchange="this.form.submit()">
                @foreach(['pending','pass','fail','na'] as $s)<option value="{{ $s }}" @selected($i->status===$s)>{{ strtoupper($s) }}</option>@endforeach
              </select>
          </td>
          <td><input class="form-control form-control-sm" name="notes" value="{{ $i->notes }}" placeholder="Notes" onchange="this.form.submit()"></td>
          <td><button class="btn btn-sm btn-outline-primary">Save</button></td>
            </form>
        </tr>
      @endforeach
      </tbody>
    </table>
  @endif
</div>
@endsection