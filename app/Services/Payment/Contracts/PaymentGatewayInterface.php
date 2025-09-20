<?php

namespace App\Services\Payment\Contracts;

use App\Models\Plan;
use App\Models\User;
use App\Services\Payment\Enums\PaymentGatewayEnum;
use Illuminate\Contracts\Auth\Authenticatable;
use Illuminate\Contracts\View\View;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Laravel\Cashier\Subscription as Subscriptions;

interface PaymentGatewayInterface
{
    public static function enum(): PaymentGatewayEnum;

    public function createCustomer(Authenticatable|User|null $user): void;

    public function saveProduct(Plan $plan): void;

    public function saveAllProducts(): void;

    public function prepaid(Plan $plan): View|RedirectResponse;

    public function subscribe(Plan $plan): View|RedirectResponse;

    public function subscribeCheckout(Request $request, ?string $referral = null): RedirectResponse;

    public function prepaidCheckout(Request $request, ?string $referral = null): RedirectResponse;

    public function handleWebhook(Request $request): JsonResponse;

    public function getSubscriptionStatus($incomingUserId = null): bool;

    public function getSubscriptionDaysLeft(): null|int|string;

    public function subscribeCancel(null|Authenticatable|User $user, ?string $msg): RedirectResponse;

    public function checkIfTrial(): bool;

    public function getSubscriptionRenewDate(): null|int|string;

    public function cancelSubscribedPlan(): bool;

    // Needed for migration extension
    public function getPlansPriceIdsForMigration(Subscriptions $subscription, ?int $planId): void;

    // Needed for migration extension
    public function getUsersCustomerIdsForMigration(Subscriptions $subscription): void;
}
