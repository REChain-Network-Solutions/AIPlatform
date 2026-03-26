@extends('customer-feedback::layouts.master')
@section('content')
<div class='card'><h1>{{ $survey->title }}</h1><p>{{ $survey->description }}</p></div><div class='card'><table><tr><th>User</th><th>Score</th><th>Feedback</th></tr>@foreach($responses as $response)<tr><td>{{ optional($response->user)->name }}</td><td>{{ $response->score }}</td><td>{{ $response->feedback }}</td></tr>@endforeach</table>{{ $responses->links() }}</div>
@endsection
