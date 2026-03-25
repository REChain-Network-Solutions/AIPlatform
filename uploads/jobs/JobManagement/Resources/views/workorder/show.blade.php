@extends('layouts.app')

@section('content')
    <!-- CONTENT WRAPPER START -->
    <div class="content-wrapper">
        @include('engineerings::workorder.ajax.show')
    </div>
    <!-- CONTENT WRAPPER END -->

@endsection
