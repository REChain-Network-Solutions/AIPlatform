<x-layout>
<h1>Expenses</h1>
<a href="{{ route('expenses.create') }}">New Expense</a>
<table><thead><tr><th>Date</th><th>Amount</th><th>Category</th><th>Description</th></tr></thead>
<tbody>
@foreach($rows as $e)
<tr><td>{{ $e->date ?? '' }}</td><td>{{ number_format($e->amount,2) }}</td><td>{{ $e->expense_category }}</td><td>{{ $e->description }}</td></tr>
@endforeach
</tbody></table>
{{ $rows->links() }}
</x-layout>


<hr>
<p>Status Legend: draft → submitted → approved → reimbursed</p>
