@extends('layouts.app')


@section('content')
@include('inspection::partials.titan-links')

    
<div class="content-wrapper">
    @include($view)
</div>

@endsection