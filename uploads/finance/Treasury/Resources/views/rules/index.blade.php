<x-layout>
  <h1>Reconciliation Rules</h1>
  <form method="POST" action="{{ route('treasury.rules.store') }}">
    @csrf
    <input name="pattern" placeholder="Pattern (LIKE match)" required>
    <input name="account_code" placeholder="Account code (e.g. 6xxx)">
    <select name="direction"><option value="out">Out</option><option value="in">In</option></select>
    <button type="submit">Add Rule</button>
  </form>
  <hr>
  <table>
    <thead><tr><th>ID</th><th>Pattern</th><th>Account</th><th>Direction</th></tr></thead>
    <tbody>
      @foreach($rules as $r)
      <tr><td>{{ $r->id }}</td><td>{{ $r->pattern }}</td><td>{{ $r->account_code }}</td><td>{{ $r->direction }}</td></tr>
      @endforeach
    </tbody>
  </table>
</x-layout>
