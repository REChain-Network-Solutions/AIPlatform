@extends('customer-feedback::layouts.master')
@section('content')
<div class='card'><h1>#{{ $ticket->id }} {{ $ticket->title }}</h1><p>{{ $ticket->description }}</p><div class='grid'><div>Status: {{ $ticket->status }}</div><div>Priority: {{ $ticket->priority }}</div><div>Type: {{ $ticket->feedback_type }}</div><div>Requester: {{ optional($ticket->requester)->name }}</div></div><p><a class='btn btn-light' href='{{ route('feedback.tickets.edit',$ticket) }}'>Edit</a></p></div>
<div class='card'><h2>Replies</h2>@forelse($ticket->replies as $reply)<div style='padding:12px 0;border-bottom:1px solid #eee'><strong>{{ optional($reply->user)->name }}</strong><div class='muted'>{{ $reply->created_at }}</div><p>{{ $reply->message }}</p></div>@empty<p>No replies yet.</p>@endforelse</div>
<div class='card'><h2>Add Reply</h2><form method='post' action='{{ route('feedback.replies.store',$ticket) }}'>@csrf<textarea name='message' rows='5'></textarea><label><input type='checkbox' name='is_internal' value='1'> Internal</label><br><br><button class='btn'>Post reply</button></form></div>
<div class='card'><h2>AI</h2><form method='post' action='{{ route('feedback.insights.analyze',$ticket) }}'>@csrf<button class='btn'>Analyze ticket</button></form></div>
@endsection
