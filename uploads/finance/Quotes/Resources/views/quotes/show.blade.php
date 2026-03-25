@extends('layouts.app')

@section('content')
<div class="container py-4">
  <div class="d-flex justify-content-between align-items-center">
    <h3>Quote {{ $quote->number }}</h3>
    <div>
      <a href="{{ route('quotes.edit', $quote->id) }}" class="btn btn-sm btn-outline-primary">Edit</a>
      <form method="post" action="{{ route('quotes.destroy', $quote->id) }}" style="display:inline">
        @csrf @method('delete')
        <button class="btn btn-sm btn-outline-danger" onclick="return confirm('Delete quote?')">Delete</button>
      </form>
    </div>
  </div>

  <div class="mt-3">
    @php
      $publicUrl = URL::temporarySignedRoute('quotes.public.show', now()->addDays(30), ['id' => $quote->id]);
    @endphp
    <div class="input-group mb-2">
      <span class="input-group-text">Public Link</span>
      <input type="text" class="form-control" value="{{ $publicUrl }}" readonly onclick="this.select()">
    </div>

    <a href="{{ route('quotes.pdf', $quote->id) }}" class="btn btn-sm btn-outline-secondary">PDF/Print</a>
    <button class="btn btn-sm btn-outline-primary" data-bs-toggle="collapse" data-bs-target="#sendForm">Send Email</button>
    <form id="sendForm" class="collapse mt-2" method="post" action="{{ route('quotes.send', $quote->id) }}">
      @csrf
      <div class="input-group">
        <input name="email" type="email" class="form-control" placeholder="customer@example.com" required>
        <button class="btn btn-success">Send</button>
      </div>
    </form>
    <form class="d-inline" method="post" action="{{ route('quotes.convert', $quote->id) }}">
      @csrf
      <button class="btn btn-sm btn-warning" onclick="return confirm('Convert to invoice?')">Convert to Invoice</button>
    </form>
  </div>

  <div class="mt-3">
    <div><strong>Client:</strong> {{ $quote->client_id ?? '-' }}</div>
    <div><strong>Status:</strong> {{ $quote->status }}</div>
    <div><strong>Valid Until:</strong> {{ optional($quote->valid_until)->toDateString() }}</div>
    <div><strong>Notes:</strong> {{ $quote->notes }}</div>
  </div>

  <h5 class="mt-4">Items</h5>
  <table class="table table-sm">
    <thead><tr><th>Description</th><th class="text-end">Qty</th><th class="text-end">Unit</th><th class="text-end">Line</th></tr></thead>
    <tbody>
      @foreach($quote->items as $it)
      <tr>
        <td>{{ $it->description }}</td>
        <td class="text-end">{{ number_format($it->qty, 2) }}</td>
        <td class="text-end">{{ number_format($it->unit_price, 2) }}</td>
        <td class="text-end">{{ number_format($it->line_total, 2) }}</td>
      </tr>
      @endforeach
    </tbody>
  </table>

  <div class="text-end">
    <div>Subtotal: {{ $quote->currency }} {{ number_format($quote->subtotal, 2) }}</div>
    <div>Tax: {{ $quote->currency }} {{ number_format($quote->tax_total, 2) }}</div>
    <h5>Grand Total: {{ $quote->currency }} {{ number_format($quote->grand_total, 2) }}</h5>
  </div>
</div>
@endsection
