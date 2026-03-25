@extends('layouts.app')

@section('content')
    <!-- CONTENT WRAPPER START -->
    <div class="content-wrapper">
        @include('engineerings::meters.ajax.show')
    </div>
    <!-- CONTENT WRAPPER END -->

@endsection
