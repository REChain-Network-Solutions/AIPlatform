@extends('customer-feedback::layouts.master')
@section('content')
<div class='card'><h1>Create NPS survey</h1><form method='post' action='{{ route('feedback.nps.store') }}'>@csrf<label>Title</label><input name='title'><label>Description</label><textarea name='description'></textarea><label>Question</label><textarea name='question'>How likely are you to recommend us to a friend or colleague?</textarea><button class='btn'>Save</button></form></div>
@endsection
