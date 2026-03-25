<x-layout>
  <h1>Expense #{{ $e->id }}</h1>
  <p>Date: {{ $e->date }} | Amount: {{ number_format($e->amount,2) }} | Status: {{ $e->status ?? 'draft' }}</p>
  <p>Description: {{ $e->description }}</p>

  <form method="POST" action="{{ route('expenses.submit', $e->id) }}" style="display:inline">@csrf<button>Submit</button></form>
  <form method="POST" action="{{ route('expenses.approve', $e->id) }}" style="display:inline">@csrf<button>Approve</button></form>
  <form method="POST" action="{{ route('expenses.reimburse', $e->id) }}" style="display:inline">@csrf<button>Reimburse</button></form>

  <h2>Receipts</h2>
  <form method="POST" action="{{ route('expenses.receipts.upload', $e->id) }}" enctype="multipart/form-data">
    @csrf
    <input type="file" name="file" required>
    <button type="submit">Upload</button>
  </form>
  <ul>
    @foreach($receipts as $r)
      <li>{{ $r->path }} ({{ $r->mime }}, {{ $r->size }} bytes)
        @if($r->ocr_text) <details><summary>OCR</summary><pre>{{ $r->ocr_text }}</pre></details> @endif
      </li>
    @endforeach
  </ul>
</x-layout>
