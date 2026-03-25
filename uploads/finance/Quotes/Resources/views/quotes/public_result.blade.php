<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quote {{ $quote->number }} – {{ ucfirst($result) }}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-4">
  <div class="card">
    <div class="card-body">
      <h3 class="mb-1">Quote {{ $quote->number }}</h3>
      @if($result === 'accepted')
        <div class="alert alert-success">Thank you. This quote has been <strong>accepted</strong>.</div>
      @isset($invoice_id)
        <div class="alert alert-info">An invoice was created automatically (ID: {{ $invoice_id }}).</div>
      @endisset
      @else
        <div class="alert alert-secondary">This quote has been <strong>rejected</strong>.</div>
      @endif
      @include('quotes::quotes._table', ['quote' => $quote])
    </div>
  </div>
</div>
</body>
</html>
