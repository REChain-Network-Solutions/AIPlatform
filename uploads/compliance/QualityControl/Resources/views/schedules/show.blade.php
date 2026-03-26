@extends('layouts.app')

@section('content')
@include('quality_control::partials.titan-links')

    <!-- CONTENT WRAPPER START -->
    <div class="content-wrapper">
        @include('quality_control::inspection_schedules.ajax.show')
    </div>
    <!-- CONTENT WRAPPER END -->

@endsection
