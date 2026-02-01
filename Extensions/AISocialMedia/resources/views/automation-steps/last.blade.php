@extends('ai-social-media::automation-steps.layout')

@section('yield_content')
	<div class="mb-8 border-b pb-6">
		<h3 class="mb-5 flex flex-wrap items-center justify-between gap-3">
			@lang('When to publish')
		</h3>
		<p>
			@lang('Choose date and time to publish your post. You can change your timezone in the profile settings.')
		</p>
	</div>

	<form
		class="flex flex-col gap-6"
		id="stepsForm"
		action="{{ route('dashboard.user.automation.step.store') }}"
		method="POST"
	>
		@csrf

		<input
			type="hidden"
			name="platform_id"
			value="{{ $platform_id }}"
		/>
		<input
			type="hidden"
			name="step"
			value="7"
		/>

		<div class="flex flex-wrap justify-between gap-12">
			<x-forms.input
				name="repeat"
				type="checkbox"
				checked
				type="checkbox"
				switcher
				label="{{ __('Repeat?') }}"
			/>

			<x-forms.input
				size="lg"
				type="select"
				name="repeat_period"
				required
			>
				<option value="day">
					{{ __('Every Day') }}
				</option>
				<option value="week">
					{{ __('Every Week') }}
				</option>
				<option value="month">
					{{ __('Every Month') }}
				</option>
			</x-forms.input>
		</div>

		<div id="date-picker">
			<x-forms.input
				id="hiddenDateInput"
				type="hidden"
				name="date"
				label="{{ __('Choose Start Day') }}"
			/>
		</div>

		<p>
			{{ __('Time') }} ({{ config('app.timezone') }}) {{ __('Current System Time:') }} <strong>{{ now()->format('H:i:s') }}</strong>
		</p>

		<x-forms.input
			class="js-time-picker"
			type="text"
			name="time"
			value="02:56"
		/>

		<x-forms.input
			class="mt-2"
			type="checkbox"
			label="{{ __('Share now for test') }}"
			name="share_test"
			type="checkbox"
			switcher
		/>

		<x-button
			variant="secondary"
			size="lg"
			type="submit"
		>
			{{ __('Next') }}
			<span class="size-7 bg-background text-foreground inline-grid place-items-center rounded-full">
            <x-tabler-chevron-right class="size-4" />
        </span>
		</x-button>
	</form>
@endsection
@push('css')
	<link
		href="{{ custom_theme_url('/assets/libs/datepicker/air-datepicker.css') }}"
		rel="stylesheet"
	/>
	<link
		href="{{ custom_theme_url('/assets/libs/picker/picker.css') }}"
		rel="stylesheet"
	/>

	<style>
		.air-datepicker {
			width: 100% !important;
		}

		.air-datepicker-cell.-selected- {
			background: #330582 !important;
		}

		.air-datepicker-cell {
			border-radius: 50% !important;
			width: 32px !important;
			justify-self: center !important;
		}

		.air-datepicker.-inline- {
			border-color: lavenderblush !important;
		}
	</style>
@endpush

@push('script')
	<script src="{{ custom_theme_url('/assets/libs/datepicker/air-datepicker.js') }}"></script>
	<script src="{{ custom_theme_url('/assets/libs/datepicker/locale/en.js') }}"></script>
	<script src="{{ custom_theme_url('/assets/libs/picker/picker.js') }}"></script>

	<script>
		$(document).ready(function() {
			"use strict";

			$('#shader-test').on('click', function() {

			});

			new AirDatepicker('#date-picker', {
				locale: defaultLocale,
				minDate: new Date(),
				onSelect({
							 date,
							 formattedDate,
							 datepicker
						 }) {
					const hiddenInput = document.getElementById('hiddenDateInput');
					hiddenInput.value = formattedDate;
				}
			});


			new Picker(document.querySelector('.js-time-picker'), {
				format: 'HH:mm',
				headers: true,
				text: {
					title: '{{ __('Pick a time') }}',
				},
			});

		});
	</script>
@endpush

