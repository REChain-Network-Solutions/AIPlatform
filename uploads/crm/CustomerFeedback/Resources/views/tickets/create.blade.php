@extends('customer-feedback::layouts.master')
@section('content')
<div class='card'><h1>Create Ticket</h1><form method='post' action='{{ route('feedback.tickets.store') }}'>@include('customer-feedback::tickets._form')</form></div>
@endsection
