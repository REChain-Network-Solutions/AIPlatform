@extends('customer-feedback::layouts.master')
@section('content')
<div class='card'><h1>Create CSAT survey</h1><form method='post' action='{{ route('feedback.csat.store') }}'>@csrf<label>Title</label><input name='title'><label>Description</label><textarea name='description'></textarea><label>Question</label><textarea name='question'>How satisfied are you with our service?</textarea><div class='grid'><div><label>Scale min</label><input name='scale_min' value='1'></div><div><label>Scale max</label><input name='scale_max' value='5'></div></div><button class='btn'>Save</button></form></div>
@endsection
