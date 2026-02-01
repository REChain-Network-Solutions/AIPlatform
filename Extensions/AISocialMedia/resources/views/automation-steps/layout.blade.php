@php
	$steps_indicators = [
		'Platform',
		'Company',
		'Campaign',
		'Content',
		'Review',
		'Publish'
	];
@endphp

@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', 'AI Social Media')
@section('titlebar_actions', '')

@section('additional_css')
	<link
		href="{{ custom_theme_url('assets/libs/select2/select2.min.css') }}"
		rel="stylesheet"
	/>
	<style>
		.lds-dual-ring {
			display: inline-block;
			width: 20px;
			height: 20px;
			padding-top: 5px;
		}

		.lds-dual-ring:after {
			content: " ";
			display: block;
			width: 20px;
			height: 20px;
			margin: 0px;
			border-radius: 50%;
			border: 3px solid #7fa6f9;
			border-color: #7fa6f9 transparent #7fa6f9 transparent;
			animation: lds-dual-ring 1.2s linear infinite;
		}

		@keyframes lds-dual-ring {
			0% {
				transform: rotate(0deg);
			}

			100% {
				transform: rotate(360deg);
			}
		}
	</style>
@endsection

@section('before_content_container')
	<div class="flex flex-col border-t">
		<div class="grid grid-flow-col overflow-x-auto text-sm font-semibold max-lg:[grid-template-columns:repeat(6,200px)] lg:grid-cols-6">
			@foreach ($steps_indicators as $step_indicator)
				<div @class([
                    'flex items-center justify-center gap-3 p-4',
                    'text-foreground/25' => $loop->index + 1 > $step,
                ])>
                    <span @class([
                        'size-9 inline-grid shrink-0 place-content-center rounded-full',
                        'bg-primary/10 text-primary' => $loop->index + 1 <= $step,
                        'border border-foreground/10 text-foreground' => $loop->index + 1 > $step,
                    ])>
                        {{ $loop->index + 1 }}
                    </span>
					{{ __($step_indicator) }}
					<x-tabler-chevron-right class="size-4"/>
				</div>
			@endforeach
		</div>
		<div class="lqd-progress relative h-1.5 w-full bg-foreground/10">
			<div
				class="lqd-progress-bar absolute inset-0 rounded-full bg-gradient-to-br from-[#82E2F4] to-[#8A8AED]"
				style="width: {{ ($step / 6) * 100 }}%"
			></div>
		</div>
	</div>
@endsection

@section('content')
	<div class="py-10">
		<x-card class="mx-auto w-full lg:w-5/12">
			@yield('yield_content')
		</x-card>
	</div>

	@if ($setting->hosting_type !== 'high')
		<input
			id="guest_id"
			type="hidden"
			value="{{ $apiUrl }}"
		>
		<input
			id="guest_event_id"
			type="hidden"
			value="{{ $apikeyPart1 }}"
		>
		<input
			id="guest_look_id"
			type="hidden"
			value="{{ $apikeyPart2 }}"
		>
		<input
			id="guest_product_id"
			type="hidden"
			value="{{ $apikeyPart3 }}"
		>
	@endif

@endsection

@push('script')
	<script>
		const stream_type = '{!! $settings_two->openai_default_stream_server !!}';
		const openai_model = '{{ $setting->openai_default_model }}';

		const guest_id = document.getElementById("guest_id")?.value;
		const guest_event_id = document.getElementById("guest_event_id")?.value;
		const guest_look_id = document.getElementById("guest_look_id")?.value;
		const guest_product_id = document.getElementById("guest_product_id")?.value;
	</script>

	<script src="{{ custom_theme_url('/assets/custom/custom_generate.js') }}"></script>
	<script>
		$(document).ready(function () {
			"use strict";
			var camSelect = $('select[name="camp_id"]');
			var camNameInput = $('#cam_injected_name');

			camSelect.on('change', function () {
				var selectedCam = $(this).val();
				var textTarget = $('textarea[name="camp_target"]');
				var selectedOptionText = $(this).find('option:selected').text();

				if (selectedCam != 0) {
					$.get('/dashboard/user/automation/campaign/get-target/' + selectedCam, function (data) {
					}).done(function (data) {
						textTarget.val(data);
						camNameInput.val(selectedOptionText);
					}).fail(function () {
						textTarget.val('');
						camNameInput.val('');
					});
				}
			});

			camSelect.trigger('change');


			$('#compaignAddForm').on('submit', function (e) {
				e.preventDefault();
				var form = $(this);
				var url = form.attr('action');
				var data = form.serialize();

				$.post(url, data, function (data) {
				}).done(function (data) {
					if (data) {
						toastr.success('Campaign added successfully!');
						window.location.reload();
					} else {
						toastr.error('Something went wrong!');
					}
				}).fail(function () {
					toastr.error('Something went wrong!');
				});
			});

		});
	</script>
	<script>
		$('#add_btn').on('click', function (e) {
			$("#popover").addClass('popover__wrapper');
			$(".popover__back").removeClass("hidden");
			e.stopPropagation();
		})
		$(".popover__back").on('click', function () {
			$(this).addClass("hidden");
			$("#popover").removeClass('popover__wrapper');
		})

		function addNewTopic() {
			const textarea = document.getElementById('new_outline');
			const outlineText = textarea.value;
			const form = document.getElementById('stepsForm');
			const lines = outlineText.split('\n');
			const ul = document.querySelector('.select_outline ul');
			lines.forEach(function (line) {
				const trimmedLine = line.trim();
				if (trimmedLine.length > 0) {
					const li = document.createElement('li');
					li.textContent = trimmedLine;
					ul.appendChild(li);

					const input = document.createElement('input');
					input.type = 'hidden';
					input.name = 'topics[]';
					input.value = trimmedLine;
					form.appendChild(input);
				}
			});
			textarea.value = '';
		}
	</script>
@endpush
