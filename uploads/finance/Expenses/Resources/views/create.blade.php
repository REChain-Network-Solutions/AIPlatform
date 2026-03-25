<x-layout>
<h1>New Expense</h1>
<form method="POST" action="{{ route('expenses.store') }}">
  @csrf
  <input type="date" name="date" required>
  <input type="number" step="0.01" name="amount" placeholder="Amount" required>
  <select name="expense_category">
    <option value="">--Category--</option>
    @foreach($cats as $c)
      <option value="{{ $c->id }}">{{ $c->name ?? ('#'.$c->id) }}</option>
    @endforeach
  </select>
  <input type="text" name="description" placeholder="Description">
  <button type="submit">Save</button>
</form>
</x-layout>
