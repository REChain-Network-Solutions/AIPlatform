@if($item['type'] !== 'bundle')

	@if($item['price'] > 0 and ! $item['installed'] and $item['licensed'] === false)
		<p class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium whitespace-nowrap">
			<span class="inline-block size-2 rounded-full bg-gray-500"></span>
			{{ __('Not Purchased') }}
		</p>
	@endif

	@if($item['price'] > 0 and ! $item['installed'] and $item['licensed'])
		<p class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium whitespace-nowrap">
			<span class="inline-block size-2 rounded-full bg-yellow-500"></span>
			{{ __('Purchased (Not Installed)') }}
		</p>
	@endif

	@if($item['price'] > 0 and $item['installed'] and $item['licensed'] and $item['version'] != $item['db_version'])
		<p class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium whitespace-nowrap">
			<span class="inline-block size-2 rounded-full bg-green-500"></span>
			{{ __('Purchased') }}
		</p>
	@endif

	@if($item['price'] > 0 and $item['installed'] and $item['licensed'] and $item['version'] == $item['db_version'])
		<p class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium whitespace-nowrap">
			<span class="inline-block size-2 rounded-full bg-green-500"></span>
			{{ __('Purchased (Installed)') }}
		</p>
	@endif

	@if(! $item['price'] and ! $item['installed'] and $item['licensed'])
		<p class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium whitespace-nowrap">
			<span class="inline-block size-2 rounded-full bg-yellow-500"></span>
			{{ __('Free (Not Installed)') }}
		</p>
	@endif

	@if(! $item['price'] and $item['installed'] and $item['licensed'] and $item['version'] != $item['db_version'])
		<p class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium whitespace-nowrap">
			<span class="inline-block size-2 rounded-full bg-green-500"></span>
			{{ __('Free') }}
		</p>
	@endif


	@if(! $item['price'] and $item['installed'] and $item['licensed'] and $item['version'] == $item['db_version'])
		<p class="mb-0 ms-3 flex items-center gap-2 text-2xs font-medium whitespace-nowrap">
			<span class="inline-block size-2 rounded-full bg-green-500"></span>
			{{ __('Free (Installed)') }}
		</p>
	@endif

@endif
