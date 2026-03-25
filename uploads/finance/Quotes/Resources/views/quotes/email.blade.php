@include('quotes::partials.brand')
<h2>Quote {{ $quote->number }}</h2>
<p>Please find your quote details below:</p>
@include('quotes::quotes._table', ['quote' => $quote])
<p>Thank you.</p>


@php
$publicUrl = URL::temporarySignedRoute('quotes.public.show', now()->addDays(30), ['id' => $quote->id]);
@endphp

<p>View this quote online: <a href="{{ $publicUrl }}">{{ $publicUrl }}</a></p>

@if(!empty(config('quotes.brand.footer_note')))
<p>{{ config('quotes.brand.footer_note') }}</p>
@endif
