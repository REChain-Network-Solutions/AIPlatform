<?php

declare(strict_types=1);

namespace App\Extensions\CheckoutRegistration\System\Http\Services\Finance;

use App\Extensions\CheckoutRegistration\System\Http\Services\Contracts\BaseGatewayService;
use App\Models\Gateways;
use App\Models\Plan;
use App\Models\User;
use Illuminate\Http\Request;

class PaypalService implements BaseGatewayService
{
    protected string $GATEWAY_CODE = 'paypal';

    protected ?Gateways $gateway;

    protected ?string $key;

    public function __construct()
    {
        $this->setGatewaysModel();
        $this->setKey();
    }

    public function createPreReqsIfNeeded(?User $user): void {}

    public function getGatewaysCode(): ?string
    {
        return $this->GATEWAY_CODE;
    }

    public function getGatewaysModel(): ?Gateways
    {
        return $this->gateway;
    }

    public function setGatewaysModel(): void
    {
        $this->gateway = Gateways::where('code', $this->GATEWAY_CODE)->where('is_active', 1)->first();
    }

    public function setKey(): void
    {
        // TODO: Implement setKey() method.
    }

    public function createSubscription(Plan $plan, User $user): array
    {
        return [];
    }

    public function createPrepaid(Plan $plan, User $user): array
    {
        return [];
    }

    public function subscribeCheckout(Request $request, $referral = null): void {}

    public function prepaidCheckout(Request $request, $referral = null): void {}

    public function checkoutData(User $user, ?int $planID): array
    {
        // TODO: Implement checkoutData() method.
        return [];
    }
}
