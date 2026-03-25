@extends('layouts.app')
@section('content')
@include('quotes::partials.brand')
<div class="container py-4">
  <h2>Quote {{ $quote->number }}</h2>
  <p>Client: {{ $quote->client_id ?? '-' }} | Currency: {{ $quote->currency }} | Valid Until: {{ optional($quote->valid_until)->toDateString() }}</p>
  @include('quotes::quotes._table', ['quote' => $quote])
</div>
@endsection
