@include('quotes::partials.brand')

<h2>Quote Accepted</h2>
<p>The following quote was accepted.</p>

<p><strong>Quote:</strong> {{ $quote->number }}<br>
<strong>Client:</strong> {{ $quote->client_id ?? '-' }}<br>
<strong>Total:</strong> {{ $quote->currency }} {{ number_format($quote->grand_total, 2) }}</p>

@include('quotes::quotes._table', ['quote' => $quote])

@isset($invoiceId)
<p><strong>Invoice created:</strong> #{{ $invoiceId }}</p>
@endisset

@if(!empty(config('quotes.brand.footer_note')))
<p class="text-muted small">{{ config('quotes.brand.footer_note') }}</p>
@endif
