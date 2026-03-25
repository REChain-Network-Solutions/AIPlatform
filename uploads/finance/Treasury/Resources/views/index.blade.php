<x-layout>
<h1>Treasury & Cash Management</h1>
<form method="POST" action="{{ route('treasury.accounts.create') }}">
  @csrf
  <input name="name" placeholder="Bank Account Name" required>
  <input name="currency" placeholder="Currency (e.g. AUD)" value="AUD">
  <input name="opening_balance" placeholder="Opening Balance" type="number" step="0.01">
  <button type="submit">Create Bank Account</button>
</form>
</x-layout>
