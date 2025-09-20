@extends('panel.layout.settings', ['disable_tblr' => true, 'layout' => 'fullwidth'])
@section('title', __('Edit Curtain Section'))
@section('titlebar_actions', '')

@section('settings')
	<form
		class="flex flex-col gap-5"
		id="item_form"
		action="{{ route('dashboard.admin.frontend.curtain.update', $item->id) }}"
		enctype="multipart/form-data"
		method="post"
	>
		@csrf
		@method('PUT')

		<x-forms.input
			id="title"
			label="{{ __('Title') }}"
			name="title"
			size="lg"
			required
			value="{{ $item->title }}"
		/>

		<x-forms.input
			tooltip="{{ __('Use html tag') }}"
			id="title_icon"
			label="{{ __('Title icon') }}"
			name="title_icon"
			size="lg"
			value="{{ $item->title_icon }}"
		/>

		<hr>

		<div class="flex justify-between items-center">
			<h3 class="text-lg font-semibold">{{ __('Sliders') }}</h3>
			<x-button
				type="button"
				id="add_slider"
				size="sm"
				variant="outline"
			>
				{{ __('Add Slider') }}
			</x-button>
		</div>

		@php
			$empty = [
			   'title'             => '',
			   'description'       => '',
			   'bg_image'          => '',
			   'bg_video'          => '',
			   'bg_color'          => '',
			   'title_color'       => '',
			   'description_color' => '',
			];

			$sliders = $item['sliders'] ?? [$empty, $empty, $empty];
		@endphp

		<div id="sliders_container">
			@foreach($sliders as $key => $slider)
				<x-card class="slider-item" data-index="{{ $key }}">
					<div class="flex justify-between items-start mb-3">
						<h4 class="text-md font-medium">{{ __('Slider') }} #<span class="slider-number">{{ $key + 1 }}</span></h4>
						<x-button
							type="button"
							class="remove_slider"
							size="xs"
							variant="ghost"
							style="color: #ef4444;"
						>
							{{ __('Remove') }}
						</x-button>
					</div>

					<div class="gap-2">
						<x-forms.input
							class="mt-2"
							type="textarea"
							id="sliders[{{ $key }}][description]"
							label="{{ __('Description') }}"
							name="sliders[{{ $key }}][description]"
							size="lg"
							value="{{ $slider['description'] ?? '' }}"
						>
							{{ $slider['description'] ?? '' }}
						</x-forms.input>
					</div>

					<div class="flex mt-3 gap-2 grid grid-cols-1 lg:grid-cols-2">
						<x-forms.input
							type="file"
							id="sliders[{{ $key }}][bg_image]"
							label="{{ __('Background Image') }}"
							name="sliders[{{ $key }}][bg_image]"
							size="lg"
							accept="image/*"
							value="{{ $slider['bg_image'] ?? '' }}"
						>
							@if(isset($slider['bg_image']) && $slider['bg_image'])
								<x-slot name="labelExtra">
									<a href="{{ $slider['bg_image'] }}" class="text-red-700">Image Download</a>
								</x-slot>
							@endif
						</x-forms.input>
						<x-forms.input
							type="file"
							id="sliders[{{ $key }}][bg_video]"
							label="{{ __('Background Video') }}"
							name="sliders[{{ $key }}][bg_video]"
							size="lg"
							accept="video/*"
							value="{{ $slider['bg_video'] ?? '' }}"
						>
							@if(isset($slider['bg_video']) && $slider['bg_video'])
								<x-slot name="labelExtra">
									<a href="{{ $slider['bg_video'] }}" class="text-red-700">Video Download</a>
								</x-slot>
							@endif
						</x-forms.input>
					</div>

					<div class="flex mt-3 gap-2 grid grid-cols-1 lg:grid-cols-3">
						<x-forms.input
							type="color"
							id="sliders[{{ $key }}][bg_color]"
							label="{{ __('Background Color') }}"
							name="sliders[{{ $key }}][bg_color]"
							size="lg"
							value="{{ $slider['bg_color'] ?? '' }}"
						/>
						<x-forms.input
							type="color"
							id="sliders[{{ $key }}][title_color]"
							label="{{ __('Title Color') }}"
							name="sliders[{{ $key }}][title_color]"
							size="lg"
							value="{{ $slider['title_color'] ?? '' }}"
						/>
						<x-forms.input
							type="color"
							id="sliders[{{ $key }}][description_color]"
							label="{{ __('Description Color') }}"
							name="sliders[{{ $key }}][description_color]"
							size="lg"
							value="{{ $slider['description_color'] ?? '' }}"
						/>
					</div>
				</x-card>
			@endforeach
		</div>

		<x-button
			id="item_button"
			size="lg"
			type="submit"
		>
			{{ __('Save') }}
		</x-button>
	</form>
@endsection

@push('script')
	<script>
		document.addEventListener('DOMContentLoaded', function() {
			const slidersContainer = document.getElementById('sliders_container');
			const addSliderBtn = document.getElementById('add_slider');

			// Get the current highest index
			function getHighestIndex() {
				const sliderItems = document.querySelectorAll('.slider-item');
				let highestIndex = -1;

				sliderItems.forEach(item => {
					const index = parseInt(item.dataset.index);
					if (index > highestIndex) {
						highestIndex = index;
					}
				});

				return highestIndex;
			}

			// Update slider numbers
			function updateSliderNumbers() {
				const sliderItems = document.querySelectorAll('.slider-item');
				sliderItems.forEach((item, index) => {
					const numberSpan = item.querySelector('.slider-number');
					if (numberSpan) {
						numberSpan.textContent = index + 1;
					}
				});
			}

			// Add new slider
			addSliderBtn.addEventListener('click', function() {
				const newIndex = getHighestIndex() + 1;

				// Clone the first existing slider and modify it
				const firstSlider = document.querySelector('.slider-item');
				const newSlider = firstSlider.cloneNode(true);

				// Update the data-index
				newSlider.dataset.index = newIndex;

				// Clear all form values
				const inputs = newSlider.querySelectorAll('input, textarea');
				inputs.forEach(input => {
					if (input.type === 'file') {
						input.value = '';
					} else if (input.type === 'color') {
						input.value = '#000000';
					} else {
						input.value = '';
					}
				});

				// Update all name and id attributes
				const formElements = newSlider.querySelectorAll('[name], [id], [for]');
				formElements.forEach(element => {
					if (element.name) {
						element.name = element.name.replace(/\[\d+\]/, `[${newIndex}]`);
					}
					if (element.id) {
						element.id = element.id.replace(/\[\d+\]/, `[${newIndex}]`);
					}
					if (element.getAttribute('for')) {
						element.setAttribute('for', element.getAttribute('for').replace(/\[\d+\]/, `[${newIndex}]`));
					}
				});

				// Remove any existing download links
				const downloadLinks = newSlider.querySelectorAll('a[href]');
				downloadLinks.forEach(link => link.remove());

				// Add to container
				slidersContainer.appendChild(newSlider);
				updateSliderNumbers();
			});

			// Remove slider
			slidersContainer.addEventListener('click', function(e) {
				if (e.target.classList.contains('remove_slider')) {
					const sliderItem = e.target.closest('.slider-item');
					const allSliders = document.querySelectorAll('.slider-item');

					// Prevent removing if only one slider remains
					if (allSliders.length > 1) {
						sliderItem.remove();
						updateSliderNumbers();
					} else {
						alert('{{ __("You must have at least one slider") }}');
					}
				}
			});
		});
	</script>
@endpush
