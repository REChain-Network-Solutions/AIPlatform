@extends('layouts.app')


@section('content')
@include('quality_control::partials.titan-links')

    
<div class="content-wrapper">
    @include($view)
</div>

@endsection