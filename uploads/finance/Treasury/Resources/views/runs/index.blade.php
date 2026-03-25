<x-layout>
  <h1>Payment Runs</h1>
  <table><thead><tr><th>ID</th><th>Scheduled</th><th>Status</th><th>Actions</th></tr></thead>
  <tbody>
    @foreach($runs as $r)
    <tr>
      <td>{{ $r->id }}</td>
      <td>{{ $r->scheduled_on }}</td>
      <td>{{ $r->status }}</td>
      <td><a href="{{ route('treasury.runs.view', $r->id) }}">View</a></td>
    </tr>
    @endforeach
  </tbody>
  </table>
  {{ $runs->links() }}
</x-layout>
