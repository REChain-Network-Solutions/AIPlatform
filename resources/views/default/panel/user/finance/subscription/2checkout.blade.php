@extends('panel.layout.app')
@section('content')
	<!-- Page body -->
	<div class="py-10">
		<div class="container-xl">
			<div class="row row-cards">
				<div class="col-sm-8 col-lg-8">
					<form
						id="payment-form"
					>
						@csrf
						<div class="row">
							<div class="col-md-12 col-xl-12">
								<!-- Button -->
								<x-button
									id="buy-button"
									class="w-full"
									type="button"
									variant="ghost-shadow"
									onclick="{{ $app_is_demo ? 'return toastr.info(\'This feature is disabled in Demo version.\')' : '' }}"
								>
									@if ($plan->trial_days != 0 && $plan->frequency != 'lifetime_monthly' && $plan->frequency != 'lifetime_yearly' && $plan->price > 0)
										<span id="button-text">
                                            {{ __('Start free trial') }}
											{{ __('with') }}
                                            <img
												class="h-auto w-24"
												src="{{ custom_theme_url('/assets/img/payments/2checkout.svg') }}"
												height="29px"
												alt="2checkout"
											>
                                        </span>
									@else
										<span id="button-text">{{ __('Pay') }}
											{!! displayCurr(currency()->symbol, $plan->price) !!}
											{{ __('with') }}
                                            <img
												class="h-auto w-24"
												src="{{ custom_theme_url('/assets/img/payments/2checkout.svg') }}"
												height="29px"
												alt="2checkout"
											>
                                        </span>
									@endif
								</x-button>
							</div>
						</div>
					</form>
					<br>
					<p>{{ __('By purchasing you confirm our') }} <a href="{{ url('/') . '/terms' }}">{{ __('Terms and Conditions') }}</a> </p>
				</div>
				<div class="col-sm-4 col-lg-4">
					@include('panel.user.finance.partials.plan_card')
				</div>
			</div>
		</div>
	</div>

	<script>
		(function (document, src, libName, config) {
			var script             = document.createElement('script');
			script.src             = src;
			script.async           = true;
			var firstScriptElement = document.getElementsByTagName('script')[0];
			script.onload          = function () {
				for (var namespace in config) {
					if (config.hasOwnProperty(namespace)) {
						window[libName].setup.setConfig(namespace, config[namespace]);
					}
				}
				window[libName].register();
			};
			firstScriptElement.parentNode.insertBefore(script, firstScriptElement);
		})(document, 'https://secure.2checkout.com/checkout/client/twoCoInlineCart.js', 'TwoCoInlineCart',{"app":{"merchant":"{{$mCode}}","iframeLoad":"checkout"},"cart":{"host":"https:\/\/secure.2checkout.com","customization":"inline-one-step"}});
	</script>

	<script>
		window.document.getElementById('buy-button').addEventListener('click', function() {
			let url = `{{ route('dashboard.user.payment.subscription.checkout', ['gateway' => ':gateway']) }}`;
			url = url.replace(':gateway', '{{\App\Services\Payment\Enums\PaymentGatewayEnum::TwoCheckout->value}}');
			TwoCoInlineCart.cart.setCurrency('USD');
			TwoCoInlineCart.products.removeAll();
			TwoCoInlineCart.products.add({
				code: "{{$product->product_id}}",
				quantity: 1,
			});
			TwoCoInlineCart.cart.setReturnMethod({
				type: 'redirect',
				url : url
			});
			// TwoCoInlineCart.cart.addCoupon('50%OFF');
			// TwoCoInlineCart.cart.setCartLockedFlag(true);
			TwoCoInlineCart.cart.checkout();
		});
	</script>
@endsection
