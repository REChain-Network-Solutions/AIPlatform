@extends('quality_control::layouts.master')

@section('content')
<div class="container-fluid">
  <div class="d-flex align-items-center justify-content-between mb-3">
    <h3 class="mb-0">{{ $template->name }}</h3>
    <div>
      <a href="{{ route('inspection-templates.edit', $template->id) }}" class="btn btn-primary">{{ __('quality_control::buttons.edit') }}</a>
      <a href="{{ route('inspection-templates.index') }}" class="btn btn-light">{{ __('quality_control::buttons.back') }}</a>
    </div>
  </div>

  <div class="card">
    <div class="card-body">
      <p class="text-muted mb-2">{{ $template->trade }}</p>
      <p>{{ $template->description }}</p>

      <hr>
      <h5>{{ __('quality_control::templates.items') }}</h5>
      <ul class="list-group">
        @foreach($template->items as $item)
          <li class="list-group-item d-flex justify-content-between">
            <span>{{ $item->item_name }}</span>
            <span class="text-muted">{{ $item->standard }}</span>
          </li>
        @endforeach
      </ul>
    </div>
  </div>
</div>
@endsection
