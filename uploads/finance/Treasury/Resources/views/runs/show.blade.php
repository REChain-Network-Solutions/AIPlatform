<x-layout>
  <h1>Payment Run #{{ $run->id }}</h1>
  <p>Scheduled: {{ $run->scheduled_on }} | Status: {{ $run->status }}</p>
  <p>
    <a href="{{ route('treasury.paymentruns.export.aba', $run->id) }}">Export ABA</a> |
    <a href="{{ route('treasury.paymentruns.export.sepa', $run->id) }}">Export SEPA</a> |
    <a href="{{ route('treasury.paymentruns.export.csv', $run->id) }}">Export CSV</a>
  </p>
  <table><thead><tr><th>Beneficiary</th><th>Amount</th><th>Reference</th></tr></thead>
  <tbody>
    @foreach($run->lines as $ln)
    <tr><td>{{ $ln->beneficiary }}</td><td>{{ number_format($ln->amount,2) }}</td><td>{{ $ln->reference }}</td></tr>
    @endforeach
  </tbody></table>
</x-layout>
