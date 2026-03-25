@extends('layouts.app')

@section('content')
    <!-- CONTENT WRAPPER START -->
    <div class="content-wrapper">
        @include('engineerings::workrequest.ajax.show')
    </div>
    <!-- CONTENT WRAPPER END -->

@endsection
