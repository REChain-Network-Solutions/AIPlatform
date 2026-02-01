@extends('panel.layout.settings')
@section('title', __('AI social media settings'))

@section('additional_css')
    <link
        rel="stylesheet"
        href="https://foliotek.github.io/Croppie/croppie.css"
    />
    <style>
        #upload-demo {
            width: 250px;
            height: 250px;
            padding-bottom: 25px;
            margin: 0 auto;
        }
    </style>
@endsection

@section('settings')
	<form
		action="{{ route('dashboard.admin.automation.settings.update') }}"
		method="POST"
	>
		@method('POST')
		@csrf
		<h3 class="mb-[25px] text-[20px]">{{ __('Instagram API Settings') }}</h3>
		<div class="row">
			<!-- TODO Serper api key -->
			<div class="col-md-12">
				<div
					class="mb-3">
					<label class="form-label">{{ __('INSTAGRAM APP ID') }}</label>
					<input
						class="form-control @error('instagram_app_id') is-invalid @enderror"
						id="instagram_app_id"
						type="text"
						name="instagram_app_id"
						value="{{ $app_is_demo ? '*********************' : old('instagram_app_id', setting('instagram_app_id')) }}"
						required
					>
					@error('instagram_app_id')
					<small class="text-red-500">{{ $message }}</small>
					@enderror
				</div>
				<div
					class="mb-3">
					<label class="form-label">{{ __('INSTAGRAM APP SECRET') }}</label>
					<input
						class="form-control @error('instagram_app_secret') is-invalid @enderror"
						id="instagram_app_secret"
						type="text"
						name="instagram_app_secret"
						value="{{ $app_is_demo ? '*********************' : old('instagram_app_secret', setting('instagram_app_secret')) }}"
						required
					>
					@error('instagram_app_secret')
					<small class="text-red-500">{{ $message }}</small>
					@enderror
				</div>
				<div
					class="mb-3">
					<label class="form-label">{{ __('INSTAGRAM REDIRECT URI') }}</label>
					<input
						disabled
						class="form-control"
						type="text"
						value="{{ url('/oauth/callback/instagram') }}"
						required
					>
				</div>
				<hr>
				<x-alert
					class="mt-2 mb-3">
					<p>
						{!! __('You can use <a class="text-red-600" href="https://developers.facebook.com/docs/instagram-platform/instagram-api-with-instagram-login/">this link</a> to access Instagram app credentials. These credentials are essential for integrating with the Instagram API and managing your application.') !!}
					</p>
				</x-alert>
			</div>
		</div>
		<button
			class="btn btn-primary w-full"
		>
			{{ __('Save') }}
		</button>
	</form>
@endsection

@push('script')

@endpush
