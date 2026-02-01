@extends('ai-social-media::automation-steps.layout')

@section('yield_content')
	<div class="mb-8 border-b pb-6">
		<h3 class="mb-5 flex flex-wrap items-center justify-between gap-3">
			@lang('Select a Platform')
			<x-button
				class="ms-auto"
				variant="secondary"
				href="{{ route('dashboard.user.automation.platform.list') }}"
			>
				{{ __('Connect Your Accounts') }}
			</x-button>
		</h3>
		<p>
			@lang('Choose the social media platforms you would like to pubish your post. Feel free to select multiple platforms at once.')
		</p>
	</div>

	<form
		class="flex flex-col gap-6"
		id="stepsForm"
		action="{{ route('dashboard.user.automation.step.second') }}"
		method="POST"
	>
		@csrf
		<input
			type="hidden"
			name="automation"
			value="{{ time() }}"
		>

		<input
			type="hidden"
			name="platform_id"
		/>
		<input
			type="hidden"
			name="step"
			value="2"
		/>
		<input
			type="hidden"
			name="automation_step"
			value="two"
		/>
		<div class="flex flex-col gap-5">

			@foreach($platforms as $platform)
				@if(! $platform->has_setting)
					@continue
				@endif

				@php
					$is_connected = $platform->setting;
					$connectionMessage = $is_connected ? __('Connected') : __('Not Connected');
					if ($is_connected) {
						if ($is_connected->expires_at && $is_connected->expires_at->isPast()) {
							$is_connected = false;
							$connectionMessage = __('Session Expired');
						}
					}
				@endphp
				<button
					class="font-sm group relative flex h-full w-full items-center gap-4 rounded-[20px] border px-3 py-2.5 font-medium transition-all group-hover:scale-110 [&.lqd-is-active]:shadow-xl [&.lqd-is-active]:shadow-black/[3%]"
					type="button"
					onclick="handleButtonClick(this, {{ $platform->id }}, {{ $is_connected }});"
				>
                <span
					class="size-9 inline-grid place-items-center rounded-full bg-foreground/10 group-[&.lqd-is-active]:bg-primary/10 group-[&.lqd-is-active]:text-primary">
                    <x-tabler-check
						class="size-5 hidden group-[&.lqd-is-active]:block"
						stroke-width="1.5"
					/>
                </span>
					{{ $platform->name }}
					<img
						class="max-w-6 max-h-6"
						src="/{{ $platform->logo }}"
					/>
					<span @class([
                    'gap-1.5 items-center flex ms-auto text-[12px]',
                    'flex' => $is_connected,
                    'hidden' => ! $is_connected,
                ])>
                    <span
						class="size-2 inline-block rounded-full bg-green-500"
						aria-hidden="true"
					></span>
                    @lang($connectionMessage)
                </span>
				</button>
			@endforeach
			<x-button
				variant="secondary"
				onclick="goNextStep();"
				type="submit"
			>
				{{ __('Next') }}
				<span class="size-7 inline-grid place-items-center rounded-full bg-background text-foreground">
                <x-tabler-chevron-right class="size-4"/>
            </span>
			</x-button>
		</div>
	</form>

@endsection

@push('script')
	<script>
		let selectedButton = null; // Initialize a variable to keep track of the selected button
		let selectedPlatformId = null;
		let is_connected_main = false;

		function handleButtonClick(button, platformId, is_connected) {
			var buttons = document.querySelectorAll('button');
			buttons.forEach(function (btn) {
				btn.classList.toggle('lqd-is-active', false);
			});

			button.classList.toggle('lqd-is-active');
			is_connected_main = is_connected;

			if (selectedButton === button) {
				// If the same button is clicked
				selectedButton = null;
				selectedPlatformId = null;

			} else {
				selectedButton = button;
				selectedPlatformId = platformId;
				// platform_id
				document.querySelector('input[name="platform_id"]').value = platformId;
			}
		}

		function goNextStep() {

			if (!selectedPlatformId) {
				event.preventDefault();
				toastr.error("{{ __('No platform selected.') }}");
			}

			if (!is_connected_main) {
				event.preventDefault();
				toastr.error("{{ __('Please connect with the platform first.') }}");
			}
		}
	</script>

@endpush
