@extends('layouts.app')

@section('content')
<div class="container py-4">
  <div class="d-flex justify-content-between align-items-center mb-3">
    <h3>Quotes & Estimates</h3>
    <a href="{{ route('quotes.create') }}" class="btn btn-primary">New Quote</a>
  </div>

  <table class="table table-striped">
    <thead>
      <tr>
        <th>#</th><th>Client</th><th>Status</th><th>Total</th><th>Valid Until</th><th></th>
      </tr>
    </thead>
    <tbody>
      @foreach($quotes as $q)
      <tr>
        <td>{{ $q->number }}</td>
        <td>{{ $q->client_id ?? '-' }}</td>
        <td>{{ $q->status }}</td>
        <td>{{ $q->currency }} {{ number_format($q->grand_total, 2) }}</td>
        <td>{{ optional($q->valid_until)->toDateString() }}</td>
        <td class="text-end">
          <a href="{{ route('quotes.show', $q->id) }}" class="btn btn-sm btn-outline-secondary">Open</a>
        </td>
      </tr>
      @endforeach
    </tbody>
  </table>

  {{ $quotes->links() }}
</div>
@endsection
