<?php

namespace App\Extensions\CheckoutRegistration\System\View\Components;

use App\Extensions\CheckoutRegistration\System\Http\Services\Finance\PaypalService;
use App\Extensions\CheckoutRegistration\System\Http\Services\Finance\StripeService;
use App\Models\Plan;
use App\Models\User;
use Closure;
use Illuminate\Contracts\Auth\Authenticatable;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class CheckoutCard extends Component
{
    /**
     * Create a new component instance.
     */
    public function __construct(
        public ?Plan $plan,
        public StripeService|PaypalService|null $gatewayService = null,
        public Authenticatable|User|null $user = null,
        public ?string $order_id = null,

        public ?float $newDiscountedPrice = null,
        public float $taxValue = 0.0,
        public array $paymentIntent = [],
        public ?string $billingPlanId = null,
        public ?int $productId = null,
    ) {
        if (! $this->plan?->exists) {
            $this->plan = Plan::where('active', 1)->first();
        }

        if (is_null($this->newDiscountedPrice) && $this->plan?->exists) {
            $this->newDiscountedPrice = $this->plan->price;
        }

        if (is_null($this->user)) {
            $this->user = auth()->check() ? auth()->user() : null;
        }

        if (! isset($this->paymentIntent['client_secret'])) {
            $this->paymentIntent['client_secret'] = 'set_your_payment_intent_client_secret';
        }
    }

    /**
     * Get the view / contents that represent the component.
     */
    public function render(): View|Closure|string
    {
        $gatewayCode = $this->gatewayService?->getGatewaysCode() ?? 'stripe';

        return view('checkout-registration::components.checkout-card-' . $gatewayCode);
    }
}
