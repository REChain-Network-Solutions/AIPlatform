@extends('panel.layout.app')
@section('title', __('Xero API'))
@section('titlebar_subtitle', '')
@section('titlebar_actions', '')

@section('content')
	<div class="py-10">
		<div class="container-xl">
			<div class="row">
				<div class="col-md-5 mx-auto">
					<form action="{{route('dashboard.admin.settings.xero.update')}}" method="POST" class="grid grid-cols-1 gap-4">
						@csrf
						<x-forms.input
							id="XERO_CLIENT_ID"
							label="{{ __('Xero Client ID') }}"
							name="XERO_CLIENT_ID"
							size="lg"
							value="{{ setting('XERO_CLIENT_ID') }}"
						/>
						<x-forms.input
							id="XERO_CLIENT_SECRET"
							label="{{ __('Xero Client Secret') }}"
							name="XERO_CLIENT_SECRET"
							size="lg"
							value="{{ setting('XERO_CLIENT_SECRET') }}"
						/>
						@php
							$websiteUrl = url('/');
						@endphp
						<x-forms.input
							id="XERO_REDIRECT_URI"
							label="{{ __('Xero Redirect URI') }}"
							name="XERO_REDIRECT_URI"
							size="lg"
							value="{{ setting('XERO_REDIRECT_URI', $websiteUrl . '/xero/connect') }}"
						/>
						<x-forms.input
							id="XERO_LANDING_URL"
							label="{{ __('Xero Landing URL') }}"
							name="XERO_LANDING_URL"
							size="lg"
							value="{{ setting('XERO_LANDING_URL', $websiteUrl . '/xero') }}"
						/>
						<button type="submit" class="btn btn-primary">{{ __('Save') }}</button>
					</form>
					<x-alert class="my-4">
						<p>
							@lang("After saving the settings, you need to click on the 'Connect to Xero' button to connect to Xero API.")
						</p>
					</x-alert>
					@if(\Dcblogdev\Xero\Facades\Xero::isConnected())
						<button type="button" class="btn btn-success">{{ __('Connected to Xero') }}</button>
					@else
						<form action="{{route('dashboard.admin.settings.xero.connect')}}" method="POST" class="grid grid-cols-1 gap-4">
							@csrf
							<button type="submit" class="btn btn-primary">{{ __('Connect to Xero') }}</button>
						</form>
					@endif
					<form action="{{route('dashboard.admin.settings.xero.create-contacts')}}" method="POST" class="grid grid-cols-1 gap-4">
						@csrf
						<x-alert class="mt-4">
							<p>
								@lang("After connecting, you need to click on the 'Create Xero Contacts' button one time to make a contact foreach exist user in the system. The upcoming contacts will be created automatically.")
							</p>
						</x-alert>
						<button type="submit" class="btn btn-primary">{{ __('Create Xero Contacts') }}</button>
					</form>
				</div>
			</div>
		</div>
	</div>
@endsection
