@extends('layouts.app')
@section('content')
<div class="container" style="max-width:700px">
  <h4>Attach Checklist to @term('work_order') #{{ $woId }}</h4>
  <form method="post" action="{{ route('workorders.checklists.attach',$woId) }}">@csrf
    <div class="mb-3">
      <label class="form-label">Template</label>
      <select name="template_id" class="form-select" required>
        @foreach($templates as $t)<option value="{{ $t->id }}">{{ $t->label }}</option>@endforeach
      </select>
    </div>
    <button class="btn btn-primary">Attach</button>
  </form>
</div>
@endsection