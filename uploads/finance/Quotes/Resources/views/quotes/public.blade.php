<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quote {{ $quote->number }}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
@include('quotes::partials.brand')
<div class="container py-4">
  <div class="card">
    <div class="card-body">
      <h3 class="mb-1">Quote {{ $quote->number }}</h3>
      <div class="text-muted mb-3">Currency: {{ $quote->currency }} &middot; Valid Until: {{ optional($quote->valid_until)->toDateString() }}</div>
      @include('quotes::quotes._table', ['quote' => $quote])

      <div class="mt-4 d-flex gap-2">
        <form method="post" action="{{ $acceptUrl }}">
          @csrf
          <button class="btn btn-success">Accept Quote</button>
        </form>
        <form method="post" action="{{ $rejectUrl }}">
          @csrf
          <button class="btn btn-outline-danger">Reject Quote</button>
        </form>
      </div>
      <p class="mt-3 text-muted small">This link is secure and uniquely signed. If it has expired, ask your contact to resend the quote.</p>
    </div>
  </div>
</div>
</body>
</html>
