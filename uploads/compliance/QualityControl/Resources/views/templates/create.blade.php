@extends('quality_control::layouts.master')

@section('content')
<div class="container-fluid">
  <h3 class="mb-3">{{ __('quality_control::templates.create_title') }}</h3>

  <div class="card">
    <div class="card-body">
      <form method="POST" action="{{ route('inspection-templates.store') }}">
        @csrf
        @include('quality_control::templates.partials._form')
        <button class="btn btn-primary">{{ __('quality_control::buttons.save') }}</button>
        <a class="btn btn-light" href="{{ route('inspection-templates.index') }}">{{ __('quality_control::buttons.cancel') }}</a>
      </form>
    </div>
  </div>
</div>
@endsection
