@extends('quality_control::layouts.master')

@section('content')
<div class="container-fluid">
  <div class="d-flex align-items-center justify-content-between mb-3">
    <h3 class="mb-0">{{ __('quality_control::templates.edit_title') }}: {{ $template->name }}</h3>
    <a href="{{ route('inspection-templates.index') }}" class="btn btn-light">{{ __('quality_control::buttons.back') }}</a>
  </div>

  @if(session('success'))
    <div class="alert alert-success">{{ session('success') }}</div>
  @endif

  <div class="row">
    <div class="col-lg-6">
      <div class="card mb-3">
        <div class="card-body">
          <form method="POST" action="{{ route('inspection-templates.update', $template->id) }}">
            @csrf
            @method('PUT')
            @include('quality_control::templates.partials._form', ['template' => $template])
            <button class="btn btn-primary">{{ __('quality_control::buttons.save') }}</button>
          </form>
        </div>
      </div>
    </div>

    <div class="col-lg-6">
      <div class="card mb-3">
        <div class="card-header">
          <strong>{{ __('quality_control::templates.items') }}</strong>
        </div>
        <div class="card-body">
          @include('quality_control::templates.partials._items', ['template' => $template])
        </div>
      </div>
    </div>
  </div>
</div>
@endsection
