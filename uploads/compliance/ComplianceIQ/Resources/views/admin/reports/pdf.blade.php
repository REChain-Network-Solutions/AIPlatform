<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Compliance Report PDF</title></head>
<body>
  <h1>{{ $report->title }}</h1>
  <p>Period: {{ $report->period_start->toDateString() }} → {{ $report->period_end->toDateString() }}</p>
  <p>Status: {{ $report->status }}</p>
  @if($report->summary)
    <h3>Summary</h3>
    <pre>{{ json_encode($report->summary, JSON_PRETTY_PRINT) }}</pre>
  @endif
</body></html>
