<x-app-layout><h1 class='text-2xl font-bold'>Work Order #{{ $order->id }}</h1><p>Status: {{ $order->status }}</p><p>Priority: {{ $order->priority }}</p><form method="POST" action="{{ route('workorders.orders.convert', $order->id) }}" class="mt-4">
  @csrf
  <button class="px-3 py-2 bg-blue-600 text-white rounded">Convert to Project</button>
</form>

</x-app-layout>