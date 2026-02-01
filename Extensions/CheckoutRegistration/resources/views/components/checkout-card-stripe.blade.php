@php
	$gateway = $gatewayService?->getGatewaysModel();
@endphp
<style>
	#payment-form {
		width: 100%;
		/* min-width: 500px; */
		align-self: center;
		box-shadow: 0px 0px 0px 0.5px rgba(50, 50, 93, 0.1),
		0px 2px 5px 0px rgba(50, 50, 93, 0.1), 0px 1px 1.5px 0px rgba(0, 0, 0, 0.07);
		border-radius: 7px;
		padding: 40px;
	}
	.hidden {
		display: none;
	}
	#payment-message {
		color: rgb(105, 115, 134);
		font-size: 16px;
		line-height: 20px;
		padding-top: 12px;
		text-align: center;
	}
	#payment-element {
		margin-bottom: 24px;
	}
	/* Buttons and links */
	button {
		background: #5469d4;
		font-family: Arial, sans-serif;
		color: #ffffff;
		border-radius: 4px;
		border: 0;
		padding: 12px 16px;
		font-size: 16px;
		font-weight: 600;
		cursor: pointer;
		display: block;
		transition: all 0.2s ease;
		box-shadow: 0px 4px 5.5px 0px rgba(0, 0, 0, 0.07);
		width: 100%;
	}
	button:hover {
		filter: contrast(115%);
	}
	button:disabled {
		opacity: 0.5;
		cursor: default;
	}
	/* spinner/processing state, errors */
	.spinner,
	.spinner:before,
	.spinner:after {
		border-radius: 50%;
	}
	.spinner {
		color: #ffffff;
		font-size: 22px;
		text-indent: -99999px;
		margin: 0px auto;
		position: relative;
		width: 20px;
		height: 20px;
		box-shadow: inset 0 0 0 2px;
		-webkit-transform: translateZ(0);
		-ms-transform: translateZ(0);
		transform: translateZ(0);
	}
	.spinner:before,
	.spinner:after {
		position: absolute;
		content: "";
	}
	.spinner:before {
		width: 10.4px;
		height: 20.4px;
		background: #5469d4;
		border-radius: 20.4px 0 0 20.4px;
		top: -0.2px;
		left: -0.2px;
		-webkit-transform-origin: 10.4px 10.2px;
		transform-origin: 10.4px 10.2px;
		-webkit-animation: loading 2s infinite ease 1.5s;
		animation: loading 2s infinite ease 1.5s;
	}
	.spinner:after {
		width: 10.4px;
		height: 10.2px;
		background: #5469d4;
		border-radius: 0 10.2px 10.2px 0;
		top: -0.1px;
		left: 10.2px;
		-webkit-transform-origin: 0px 10.2px;
		transform-origin: 0px 10.2px;
		-webkit-animation: loading 2s infinite ease;
		animation: loading 2s infinite ease;
	}
	@-webkit-keyframes loading {
		0% {
			-webkit-transform: rotate(0deg);
			transform: rotate(0deg);
		}
		100% {
			-webkit-transform: rotate(360deg);
			transform: rotate(360deg);
		}
	}
	@keyframes loading {
		0% {
			-webkit-transform: rotate(0deg);
			transform: rotate(0deg);
		}
		100% {
			-webkit-transform: rotate(360deg);
			transform: rotate(360deg);
		}
	}
	@media only screen and (max-width: 600px) {
		form {
			width: 80vw;
			min-width: initial;
		}
	}
</style>
<form
	id="payment-form"
	action="{{ route('dashboard.user.payment.subscription.checkout', ['gateway' => 'stripe']) }}"
	method="post"
>
	@csrf
	<input type="hidden" name="planID" value="{{ $plan->id }}">
	<input type="hidden" name="couponID" id="coupon">
	{{--    <input type="hidden" name="orderID" value="{{$order_id}}">--}}
	<input type="hidden" name="payment_method" class="payment-method">
	<input type="hidden" name="gateway" value="stripe">
	<div class="row">
		<div class="col-md-12 col-xl-12">
			<x-forms.input
				label="{{ __('Email Address') }}"
				class="form-control mb-5"
				id="email"
				type="email"
				name="email"
				required
			/>
			<div id="payment-element">

			</div>
			<x-button
				class="w-full rounded-md"
				id="submit"
				type="submit"
			>
				<div
					class="spinner hidden"
					id="spinner"
				></div>
				<span id="button-text">
                    @if ($plan->trial_days !== 0 && $plan->frequency !== 'lifetime_monthly' && $plan->frequency !== 'lifetime_yearly' && $plan->price > 0)
						{{ __('Start free trial') }}
					@else
						{{ __('Pay') }}
						{!! displayCurr(currency()->symbol, $plan->price, $taxValue, $newDiscountedPrice) !!}
					@endif
					{{ __('with') }}
                     <img
						 class="h-6 w-24"
						 src="{{ custom_theme_url('/images/payment/stripe.svg') }}"
						 alt="Stripe"
					 >
                </span>
			</x-button>
			<div
				class="hidden"
				id="payment-message"
			></div>
		</div>
	</div>
</form>
<br>
<p>{{ __('By purchasing you confirm our') }} <a href="{{ url('/') . '/terms' }}">{{ __('Terms and Conditions') }}</a> </p>
<script src="{{ custom_theme_url('https://js.stripe.com/v3/') }}"></script>
<script>
	document.querySelector("#payment-form").addEventListener("submit", handlePaymentFormSubmit);
	let isRegistering = false;
	async function handlePaymentFormSubmit(event) {
		event.preventDefault(); // Prevent default form submission.
		if (isRegistering) {
			return;
		}
		isRegistering = true;
		const emailInput = document.querySelector("#email");
		const email = emailInput.value.trim();
		if (!email || !email.includes("@")) {
			displayError(emailInput, "Please enter a valid email address.");
			isRegistering = false;
			return;
		}
		clearError(emailInput);
		setLoading(true);
		try {
			const response = await fetch("{{ route('register-user') }}", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					"X-CSRF-TOKEN": "{{ csrf_token() }}"
				},
				body: JSON.stringify({ email, planID: "{{ $plan->id }}" })
			});
			if (!response.ok) {
				const error = await response.json();
				isRegistering = false;
				throw new Error(error.message || "Something went wrong. Please try again.");
			}
			const { checkoutData } = await response.json();
			if (checkoutData?.paymentIntent) {
				const stripe = Stripe("{{ $gateway->mode === 'live' ? $gateway->live_client_id : $gateway->sandbox_client_id }}");
				await initialize(stripe, checkoutData.paymentIntent);
			}
		} catch (error) {
			showMessage(error.message || "An error occurred during payment.");
		} finally {
			setLoading(false);
		}
	}
	function displayError(input, message) {
		let errorElement = input.nextElementSibling;
		if (!errorElement || !errorElement.classList.contains("error-message")) {
			errorElement = document.createElement("div");
			errorElement.className = "error-message";
			input.parentNode.appendChild(errorElement);
		}
		errorElement.textContent = message;
		errorElement.style.color = "red";
	}
	function clearError(input) {
		const errorElement = input.nextElementSibling;
		if (errorElement && errorElement.classList.contains("error-message")) {
			errorElement.textContent = "";
		}
	}
	async function initialize(stripe, paymentIntent) {
		const elements = stripe.elements({ clientSecret: paymentIntent.client_secret });
		const paymentElementOptions = {
			layout: "tabs",
			business: {
				name: "{{ config('app.name') }}"
			}
		};
		const paymentElement = elements.create("payment", paymentElementOptions);
		paymentElement.mount("#payment-element");
		if (!paymentIntent.client_secret.startsWith("set")) {
			await checkStatus(stripe, paymentIntent.client_secret);
		}
		document.querySelector("#payment-form").addEventListener("submit", (e) =>
			handleSubmit(e, stripe, paymentIntent, elements)
		);
	}
	async function handleSubmit(e, stripe, paymentIntent, elements, user) {
		e.preventDefault();
		setLoading(true); // Start spinner before submitting the payment.
		const confirmFunction = paymentIntent.client_secret.startsWith("set")
			? stripe.confirmSetup
			: stripe.confirmPayment;
		try {
			const subsUrl = `{{ route('register.checkout', ['type' => 'subscribe']) }}`;
			const prepUrl = `{{ route('register.checkout', ['type' => 'prepaid']) }}`;
			const { error } = await confirmFunction({
				elements,
				confirmParams: {
					return_url: paymentIntent.type === "subscription" ? subsUrl : prepUrl,
					payment_method_data: {
						billing_details: {
							name: user?.name,
							email: user?.email
						},
						metadata: {
							userId: user?.id,
							planId: {{ $plan->id }},
						}
					}
				}
			});
			if (error) throw error;
		} catch (error) {
			showMessage(error.message || "An error occurred during payment.");
			isRegistering = false;
		} finally {
			setLoading(false);
			isRegistering = false;
		}
	}
	async function checkStatus(stripe, clientSecret) {
		if (!clientSecret) return;
		const { paymentIntent } = await stripe.retrievePaymentIntent(clientSecret);
		switch (paymentIntent.status) {
		case "succeeded":
			showMessage("Payment succeeded!");
			break;
		case "processing":
			showMessage("Your payment is processing.");
			break;
		case "requires_payment_method":
			showMessage("Please select a valid payment method.");
			break;
		default:
			showMessage("Payment status unknown.");
		}
	}
	function setLoading(isLoading) {
		const submitButton = document.querySelector("#submit");
		const spinner = document.querySelector("#spinner");
		const buttonText = document.querySelector("#button-text");
		submitButton.disabled = isLoading;
		if (spinner) spinner.classList.toggle("hidden", !isLoading);
		if (buttonText) buttonText.classList.toggle("hidden", isLoading);
	}
	function showMessage(messageText) {
		const messageContainer = document.querySelector("#payment-message");
		if (!messageContainer) return;
		messageContainer.classList.remove("hidden");
		messageContainer.textContent = messageText;
		setTimeout(() => {
			messageContainer.classList.add("hidden");
			messageContainer.textContent = "";
		}, 7000);
	}
</script>
