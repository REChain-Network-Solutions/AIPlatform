@extends('customer-feedback::layouts.master')
@section('content')
<div class='card'><h1>NPS Analytics</h1><p>NPS score: {{ round($npsScore ?? 0,2) }}</p><p>Promoters: {{ $promoters ?? 0 }} | Passives: {{ $passives ?? 0 }} | Detractors: {{ $detractors ?? 0 }}</p></div><div class='card'><table><tr><th>Score</th><th>Feedback</th></tr>@foreach($feedback as $item)<tr><td>{{ $item->score }}</td><td>{{ $item->feedback }}</td></tr>@endforeach</table></div>
@endsection
