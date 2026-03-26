@extends('layouts.app')

@section('content')
@include('inspection::partials.titan-links')

    <!-- CONTENT WRAPPER START -->
    <div class="content-wrapper">
        @include('inspection::inspection_schedules.ajax.show')
    </div>
    <!-- CONTENT WRAPPER END -->

@endsection
