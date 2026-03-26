@extends('customer-feedback::layouts.master')
@section('content')
<div class='card'><h1>Insights dashboard</h1></div><div class='card'><h2>Insight summary</h2><table><tr><th>Type</th><th>Count</th></tr>@foreach($insightSummary as $item)<tr><td>{{ $item->insight_type }}</td><td>{{ $item->count }}</td></tr>@endforeach</table></div><div class='card'><h2>Recent insights</h2>@foreach($recentInsights as $insight)<div style='padding:10px 0;border-bottom:1px solid #eee'><strong>{{ $insight->title }}</strong> ({{ $insight->insight_type }})<br>{{ $insight->description }}</div>@endforeach</div>
@endsection
