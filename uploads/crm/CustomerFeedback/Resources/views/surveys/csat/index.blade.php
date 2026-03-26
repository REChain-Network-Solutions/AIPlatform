@extends('customer-feedback::layouts.master')
@section('content')
<div class='card'><h1>CSAT surveys</h1><a class='btn' href='{{ route('feedback.csat.create') }}'>Create CSAT survey</a></div><div class='card'><table><tr><th>Title</th><th>Responses</th><th></th></tr>@foreach($surveys as $survey)<tr><td>{{ $survey->title }}</td><td>{{ $survey->responses_count }}</td><td><a href='{{ route('feedback.csat.show',$survey) }}'>Open</a></td></tr>@endforeach</table>{{ $surveys->links() }}</div>
@endsection
