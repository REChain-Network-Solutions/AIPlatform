@php
	$item->photo = str_starts_with($item->photo, 'http')
		? $item->photo
		: asset('uploads/' . $item->photo);

	$dataIndex = isset($loop) ? $loop->index : rand(10000, 90000);
@endphp

@if ($item->status == 'in_progress')
	<x-card
		id="video-{{ $item->request_id }}"
		class="image-result group flex text-center shadow-[0_2px_2px_hsla(0,0%,0%,0.07)]"
		class:body="flex flex-col grow p-9"
		data-index="{{ $loop->index }}"
		x-ref="image-result-{{ $loop->index }}"
	>
		<div class="px-5 py-9">
			<svg
				class="size-7 mx-auto mb-3 animate-spin"
				width="28"
				height="28"
				viewBox="0 0 28 28"
				fill="none"
				xmlns="http://www.w3.org/2000/svg"
			>
				<path
					d="M14.0013 27.3333C6.65464 27.3333 0.667969 21.3467 0.667969 14C0.667969 11.5067 1.3613 9.08 2.66797 6.97333C3.05464 6.34667 3.8813 6.16 4.50797 6.54667C5.13464 6.93333 5.3213 7.75999 4.93464 8.38665C3.89464 10.0667 3.33464 12.0133 3.33464 14C3.33464 19.88 8.1213 24.6667 14.0013 24.6667C19.8813 24.6667 24.668 19.88 24.668 14C24.668 8.12 19.8813 3.33333 14.0013 3.33333C13.268 3.33333 12.668 2.73333 12.668 2C12.668 1.26667 13.268 0.666666 14.0013 0.666666C21.348 0.666666 27.3346 6.65333 27.3346 14C27.3346 21.3467 21.348 27.3333 14.0013 27.3333Z"
					fill="url(#loader-spinner-gradient)"
				/>
			</svg>
			<span
				class="inline-block bg-gradient-to-r from-[#82E2F4] to-[#6977DE] bg-clip-text text-sm font-semibold text-transparent">
                        @lang('In Progress')
                    </span>
		</div>
	</x-card>
@else
	<div
		class="image-result lqd-loading-skeleton group w-full"
		id="video-{{ $item->id }}"
		data-id="{{ $item->id }}"
		data-generator="{{ strtolower($item->response) }}"
	>
		<figure
			class="lqd-image-result-fig relative mb-3 aspect-square overflow-hidden rounded-lg shadow-md transition-all group-hover:-translate-y-1 group-hover:scale-105 group-hover:shadow-lg"
		>
			@if ($item->status == 'in_progress')
				<div class="flex items-center justify-center h-full">
					<svg
						class="size-7 mx-auto mb-3 animate-spin"
						width="28"
						height="28"
						viewBox="0 0 28 28"
						fill="none"
						xmlns="http://www.w3.org/2000/svg"
					>
						<path
							d="M14.0013 27.3333C6.65464 27.3333 0.667969 21.3467 0.667969 14C0.667969 11.5067 1.3613 9.08 2.66797 6.97333C3.05464 6.34667 3.8813 6.16 4.50797 6.54667C5.13464 6.93333 5.3213 7.75999 4.93464 8.38665C3.89464 10.0667 3.33464 12.0133 3.33464 14C3.33464 19.88 8.1213 24.6667 14.0013 24.6667C19.8813 24.6667 24.668 19.88 24.668 14C24.668 8.12 19.8813 3.33333 14.0013 3.33333C13.268 3.33333 12.668 2.73333 12.668 2C12.668 1.26667 13.268 0.666666 14.0013 0.666666C21.348 0.666666 27.3346 6.65333 27.3346 14C27.3346 21.3467 21.348 27.3333 14.0013 27.3333Z"
							fill="url(#loader-spinner-gradient)"
						/>
					</svg>
					<span class="text-sm font-semibold text-gray-600">@lang('In Progress')</span>
				</div>
			@else
				<img
					class="lqd-image-result-img aspect-square h-full w-full object-cover object-center"
					loading="lazy"
					src="{{ $item->photo }}"
				>
				<div
					class="lqd-image-result-actions absolute inset-0 flex w-full flex-col items-center justify-center gap-2 p-4 transition-opacity group-hover:opacity-100"
				>
					<x-button
						class="lqd-image-result-download download size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white"
						size="none"
						href="{{ $item->photo }}"
						download="{{ $item->photo }}"
					>
						<x-tabler-download class="size-5"/>
					</x-button>
					<x-button
						class="lqd-image-result-view gallery size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white"
						data-payload="{{ $item }}"
						@click.prevent="setActiveItem(JSON.parse($el.getAttribute('data-payload') || {})); modalShow = true"
						size="none"
						href="#"
					>
						<x-tabler-eye class="size-5"/>
					</x-button>
					<x-button
						class="lqd-image-result-delete delete size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-red-500 hover:text-white"
						size="none"
						onclick="return confirm('Are you sure?')"
						href="{{ route('dashboard.user.photo-studio.delete', $item->id) }}"
					>
						<x-tabler-x class="size-4"/>
					</x-button>
				</div>
			@endif
		</figure>
		<p class="lqd-image-result-title mb-1 w-full overflow-hidden overflow-ellipsis whitespace-nowrap text-heading-foreground">
			{{ $item->input }}
		</p>
	</div>

@endif
