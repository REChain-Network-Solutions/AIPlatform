@extends('inspection::layouts.master')

@section('content')
<div class="container-fluid">
  <div class="d-flex align-items-center justify-content-between mb-3">
    <h3 class="mb-0">{{ __('inspection::templates.title') }}</h3>
    <a href="{{ route('inspection-templates.create') }}" class="btn btn-primary">
      {{ __('inspection::templates.create') }}
    </a>
  </div>

  @if(session('success'))
    <div class="alert alert-success">{{ session('success') }}</div>
  @endif

  <div class="card">
    <div class="card-body">
      @if($templates->count() === 0)
        @include('inspection::templates.partials._empty')
      @else
        @include('inspection::templates.partials._table', ['templates' => $templates])
      @endif
    </div>
  </div>

  <div class="mt-3">
    {{ $templates->links() }}
  </div>
</div>
@endsection
