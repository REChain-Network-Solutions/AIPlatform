@extends('inspection::layouts.master')

@section('content')
<div class="container-fluid">
  <div class="d-flex align-items-center justify-content-between mb-3">
    <h3 class="mb-0">{{ __('inspection::templates.edit_title') }}: {{ $template->name }}</h3>
    <a href="{{ route('inspection-templates.index') }}" class="btn btn-light">{{ __('inspection::buttons.back') }}</a>
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
            @include('inspection::templates.partials._form', ['template' => $template])
            <button class="btn btn-primary">{{ __('inspection::buttons.save') }}</button>
          </form>
        </div>
      </div>
    </div>

    <div class="col-lg-6">
      <div class="card mb-3">
        <div class="card-header">
          <strong>{{ __('inspection::templates.items') }}</strong>
        </div>
        <div class="card-body">
          @include('inspection::templates.partials._items', ['template' => $template])
        </div>
      </div>
    </div>
  </div>
</div>
@endsection
