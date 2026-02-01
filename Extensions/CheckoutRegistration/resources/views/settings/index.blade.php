@extends('panel.layout.settings')
@section('title', __('Checkout Registration Settings'))
@section('titlebar_actions', '')
@section('settings')
<form action="{{route("dashboard.admin.checkout.registration.settings.store")}}" method="POST">
@csrf
<div class="col-md-12">
	<div class="mb-4">
		<label class="form-label">
			{{ __('Checkout Registration Status') }}
			<x-info-tooltip text="{{ __('If this is enabled users can register during checkout.') }}"/>
		</label>
		<select
			class="form-select"
			id="checkout_registration_status"
			name="checkout_registration_status"
		>
			<option
				value="active"
				{{ setting('checkout_registration_status', 'passive') === 'active' ? 'selected' : '' }}
			>
			{{ __('Active') }}</option>
			<option
				value="passive"
				{{ setting('checkout_registration_status', 'passive') === 'passive' ? 'selected' : '' }}
			>
			{{ __('Passive') }}</option>
		</select>
	</div>
</div>
<div class="col-md-12">
	<div class="mb-4">
		<label class="form-label">
			{{ __('Checkout Registration Payment Gateway') }}
			<x-info-tooltip text="{{ __('Please select the payment gateway that will be used for registration. (only Stripe is available for now.') }}"/>
		</label>
		<select
			class="form-select"
			id="default_checkout_gateway"
			name="default_checkout_gateway"
		>
			<option
				value="stripe"
				{{ setting('default_checkout_gateway', 'stripe') === 'stripe' ? 'selected' : '' }}
			>
			{{ __('Stripe') }}</option>
{{--			<option--}}
{{--				value="paypal"--}}
{{--				{{ setting('default_checkout_gateway', 'stripe') === 'paypal' ? 'selected' : '' }}--}}
{{--			>--}}
{{--			{{ __('Paypal') }}</option>--}}
		</select>
	</div>
</div>

<div class="col-md-12">
	<div class="mb-4">
		<label class="form-label">
			{{ __('Default Checkout Plan') }}
			<x-info-tooltip text="{{ __('Please select the default plan that will be used for registration.') }}"/>
		</label>
		<select
			class="form-select"
			id="default_checkout_plan_id"
			name="default_checkout_plan_id"
		>
			@foreach($plans as $plan)
			<option
				value="{{ $plan->id }}"
				{{ setting('default_checkout_plan_id', 0) == $plan->id ? 'selected' : '' }}
			>
			{{ $plan->name }}</option>
			@endforeach
		</select>
	</div>
</div>

<div class="col-12 mt-4">
	<button
		class="btn btn-primary w-full"
		type="submit"
	>
		{{ __('Save') }}
	</button>
</div>
</form>
@endsection
