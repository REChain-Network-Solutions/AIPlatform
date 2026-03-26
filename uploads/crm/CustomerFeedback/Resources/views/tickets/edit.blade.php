@extends('customer-feedback::layouts.master')
@section('content')
<div class='card'><h1>Edit Ticket</h1><form method='post' action='{{ route('feedback.tickets.update',$ticket) }}'>@csrf @method('PUT')@include('customer-feedback::tickets._form')</form></div>
@endsection
