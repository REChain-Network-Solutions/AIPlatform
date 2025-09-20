<x-slot name="header">
    <h2 class="font-semibold text-xl text-gray-800 leading-tight">
        Extension Installation Results
    </h2>
</x-slot>

@if (count($results) > 0)
    <ul>
        @foreach ($results as $slug => $result)
            <li class="mb-4">
                <strong>{{ ucfirst(str_replace('-', ' ', $slug)) }}:</strong>
                <span class="{{ $result['status'] === 'success' ? 'text-green-500' : 'text-red-500' }}">
                    {{ $result['message'] }}
                </span>
            </li>
        @endforeach
    </ul>
@else
    <p>No extensions were processed.</p>
@endif
<a href="{{ url('/dashboard/extensions') }}" class="inline-flex items-center px-4 py-2 bg-gray-800 border border-transparent rounded-md font-semibold text-xs text-white uppercase tracking-widest hover:bg-gray-700 active:bg-gray-900 focus:outline-none focus:border-gray-900 focus:ring focus:ring-gray-300 disabled:opacity-25 transition ease-in-out duration-150">
    Back to Extensions
</a>