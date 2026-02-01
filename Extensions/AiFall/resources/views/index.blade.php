@php
	$actions = [
		'luma-dream-machine' => 'Generate video with Luma',
		'kling' => 'Generate video with Kling',
		'runway-gen3' => 'Generate video with image',
		'minimax' => 'Generate video with Minimax',
	];
@endphp

@extends('panel.layout.app')
@section('title', __('AI Video Pro'))
@section('titlebar_subtitle', __('You can create amazing videos with AI Video Pro'))

@section('content')
	<div class="py-10">
		<div class="flex flex-wrap justify-between gap-y-8" x-data="{ selectedAction: '' }">
			<form
				class="w-full lg:w-4/12"
				id="photo-studio-form"
				method="post"
				action="{{ route('dashboard.user.fall-video.store') }}"
				enctype="multipart/form-data"
			>
				@csrf

				<x-forms.input
					id="action"
					name="action"
					label="{{ __('Choose an Action') }}"
					size="lg"
					type="select"
					x-model="selectedAction"
					class="mb-4"
				>
					@foreach ($actions as $value => $label)
						<option
							value="{{ $value }}"
							@selected($loop->first)
						>
							{{ __($label) }}
						</option>
					@endforeach
				</x-forms.input>

				<div class="space-y-5">

					<div
						x-show="selectedAction == 'runway-gen3'"
						class="flex w-full items-center justify-center"
						id="img_select"
						ondrop="dropHandler(event, 'img2img_src');"
						ondragover="dragOverHandler(event);"
					>
						<label
							class="lqd-filepicker-label min-h-36 mb-5 flex w-full cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-foreground/10 bg-background p-6 text-center text-xs font-medium transition-colors hover:bg-background/80"
							for="img2img_src"
						>
							<div class="flex flex-col items-center justify-center">
								<x-tabler-camera-plus
									class="size-5 mb-4"
									stroke-width="1.5"
								/>

								<p class="mb-0 opacity-50">
									{{ __('Drag and drop a source image') }}
								</p>

								<p class="file-name mb-2 text-2xs">
									{{ __('or click here to browse your files.') }}
								</p>

								<p class="mb-0 text-3xs opacity-50">
									{{ __('(Max file size: 5MB)') }}
								</p>
							</div>

							<input
								class="hidden"
								id="img2img_src"
								name="photo"
								type="file"
								accept=".png, .jpg, .jpeg"
								onchange="handleFileSelect('img2img_src')"
							/>
						</label>
					</div>

					<x-forms.input
						id="prompt"
						name="prompt"
						label="{{ __('Prompt') }}"
						size="lg"
						rows="4"
						type="textarea"
					/>

					<x-button
						class="mt-4 w-full btn_loading"
						size="lg"
						type="submit"
					>
						@lang('Generate')
					</x-button>
				</div>
			</form>

		</div>

		<div
			class="lqd-ai-videos-wrap"
			id="lqd-ai-videos-wrap"
		>
			<svg
				width="0"
				height="0"
			>
				<defs>
					<linearGradient
						id="loader-spinner-gradient"
						x1="0.667969"
						y1="6.10667"
						x2="23.0413"
						y2="25.84"
						gradientUnits="userSpaceOnUse"
					>
						<stop stop-color="#82E2F4" />
						<stop
							offset="0.502"
							stop-color="#8A8AED"
						/>
						<stop
							offset="1"
							stop-color="#6977DE"
						/>
					</linearGradient>
				</defs>
			</svg>

			@if (filled($list))
				<h3 class="my-8">
					@lang('My Videos')
				</h3>
			@else
				<h2 class="col-span-full flex items-center justify-center">
					@lang('No videos found.')
				</h2>
			@endif

			<div id="videos-container">
				@include('ai-fall-video::videos-list', ['list' => $list])
			</div>

		</div>

	</div>

	<template id="image_result">
		<div class="image-result lqd-loading-skeleton lqd-is-loading group w-full">
			<figure
				class="lqd-image-result-fig relative mb-3 aspect-square overflow-hidden rounded-lg shadow-md transition-all group-hover:-translate-y-1 group-hover:scale-105 group-hover:shadow-lg"
				data-lqd-skeleton-el
			>
				<img
					class="lqd-image-result-img aspect-square h-full w-full object-cover object-center"
					loading="lazy"
					src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgc3R5bGU9ImZpbGw6I2VlZWVlZTsiLz48L3N2Zz4="
				>
				<div
					class="lqd-image-result-actions absolute inset-0 flex w-full flex-col items-center justify-center gap-2 p-4 transition-opacity group-[&.lqd-is-loading]:invisible group-[&.lqd-is-loading]:opacity-0">
					<div class="opacity-0 transition-opacity group-hover:opacity-100">
						<x-button
							class="lqd-image-result-download download size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white"
							size="none"
							href="#"
							download=true
						>
							<x-tabler-download class="size-5"/>
						</x-button>
						<x-button
							class="lqd-image-result-view gallery size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-emerald-400 hover:text-white"
							@click.prevent="setActiveItem( JSON.parse($el.getAttribute('data-payload') || {}) ); modalShow = true"
							size="none"
							href="#"
						>
							<x-tabler-eye class="size-5"/>
						</x-button>
						<x-button
							class="lqd-image-result-delete delete size-9 rounded-full bg-background text-foreground hover:bg-background hover:bg-red-500 hover:text-white"
							size="none"
							onclick="return confirm('Are you sure?')"
							href="#"
						>
							<x-tabler-x class="size-4"/>
						</x-button>
					</div>
					<span
						class="lqd-image-result-type absolute bottom-4 end-4 mb-0 rounded-full bg-background px-2 py-1 text-3xs font-semibold uppercase leading-none transition-opacity group-[&.lqd-is-loading]:invisible group-[&[data-generator=de]]:text-red-500 group-[&[data-generator=sd]]:text-blue-500 group-[&.lqd-is-loading]:opacity-0"
					></span>
				</div>
			</figure>
			<p
				class="lqd-image-result-title mb-1 w-full overflow-hidden overflow-ellipsis whitespace-nowrap text-heading-foreground transition-opacity"
				data-lqd-skeleton-el
			></p>
		</div>
	</template>
@endsection

@push('script')
	<script>
		function checkVideoStatus() {
			fetch('{!! route('dashboard.user.fall-video.check', ['ids' => $inProgress]) !!}')
				.then(response => response.json())
				.then(data => {
					for (const [id, item] of Object.entries(data.data)) {
						let videoElement = document.getElementById(item.divId);
						if (videoElement) {
							videoElement.innerHTML = item.html;
						}
					}
				})
				.catch(error => console.error('Error:', error));
		}

		document.addEventListener('DOMContentLoaded', function () {
			setInterval(checkVideoStatus, 5000);
		});
	</script>
@endpush
