@extends('customer-feedback::layouts.master')
@section('content')
<div class='card'><h1>CSAT Analytics</h1><p>Average score: {{ round($averageScore ?? 0,2) }}</p><p>Responses: {{ $responses ?? 0 }} | Satisfaction rate: {{ $satisfactionRate ?? 0 }}%</p></div><div class='card'><table><tr><th>Score</th><th>Count</th></tr>@foreach($scoreDistribution as $item)<tr><td>{{ $item->score }}</td><td>{{ $item->count }}</td></tr>@endforeach</table></div>
@endsection
