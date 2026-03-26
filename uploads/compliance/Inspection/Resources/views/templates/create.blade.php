@extends('inspection::layouts.master')

@section('content')
<div class="container-fluid">
  <h3 class="mb-3">{{ __('inspection::templates.create_title') }}</h3>

  <div class="card">
    <div class="card-body">
      <form method="POST" action="{{ route('inspection-templates.store') }}">
        @csrf
        @include('inspection::templates.partials._form')
        <button class="btn btn-primary">{{ __('inspection::buttons.save') }}</button>
        <a class="btn btn-light" href="{{ route('inspection-templates.index') }}">{{ __('inspection::buttons.cancel') }}</a>
      </form>
    </div>
  </div>
</div>
@endsection
