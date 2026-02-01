@if($item['licensed'] === false)
	<p class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium">
		<span class="inline-block size-2 rounded-full bg-gray-500"></span>
		{{ __('Not Purchased') }}
	</p>
@endif

@if($item['licensed'])
	<p class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium">
		<span class="inline-block size-2 rounded-full bg-yellow-500"></span>
		{{ __('Purchased') }}
	</p>
@endif

