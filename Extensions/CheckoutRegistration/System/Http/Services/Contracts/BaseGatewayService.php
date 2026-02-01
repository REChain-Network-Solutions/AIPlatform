<?php

namespace App\Extensions\CheckoutRegistration\System\Http\Services\Contracts;

use App\Models\Gateways;
use App\Models\Plan;
use App\Models\User;
use Illuminate\Http\Request;

interface BaseGatewayService
{
    public function createPreReqsIfNeeded(?User $user): void;

    public function setGatewaysModel(): void;

    public function getGatewaysCode(): ?string;

    public function getGatewaysModel(): ?Gateways;

    public function setKey(): void;

    public function checkoutData(User $user, ?int $planID): array;

    public function createSubscription(Plan $plan, User $user): array;

    public function createPrepaid(Plan $plan, User $user): array;

    public function subscribeCheckout(Request $request, $referral = null);

    public function prepaidCheckout(Request $request, $referral = null);
}
