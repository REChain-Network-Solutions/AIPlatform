<?php

declare(strict_types=1);

namespace App\Extensions\CheckoutRegistration\System\Http\Services\Finance;

use App\Actions\CreateActivity;
use App\Enums\Plan\FrequencyEnum;
use App\Enums\Plan\TypeEnum;
use App\Extensions\Affilate\System\Events\AffiliateEvent;
use App\Extensions\CheckoutRegistration\System\Http\Services\Contracts\BaseGatewayService;
use App\Helpers\Classes\Helper;
use App\Jobs\CancelAwaitingPaymentSubscriptions;
use App\Models\Coupon;
use App\Models\Currency;
use App\Models\GatewayProducts;
use App\Models\Gateways;
use App\Models\Plan;
use App\Models\Setting;
use App\Models\User;
use App\Models\UserOrder;
use App\Services\PaymentGateways\Contracts\CreditUpdater;
use Exception;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;
use Laravel\Cashier\Subscription as Subscriptions;
use RuntimeException;
use Stripe\Exception\ApiErrorException;
use Stripe\PaymentIntent;
use Stripe\Stripe;
use Stripe\StripeClient;

class StripeService implements BaseGatewayService
{
    use CreditUpdater;

    protected string $GATEWAY_CODE = 'stripe';

    protected ?Gateways $gateway;

    protected ?string $key;

    protected ?StripeClient $stripe_client;

    public function __construct()
    {
        $this->setGatewaysModel();
        $this->setKey();
        $this->setClient();
    }

    public function createPreReqsIfNeeded(?User $user): void
    {
        if ($user?->stripe_id === null) {
            $this->createStripeCustomer($user);
        }
    }

    public function getGatewaysCode(): ?string
    {
        return $this->GATEWAY_CODE;
    }

    public function setGatewaysModel(): void
    {
        $this->gateway = Gateways::where('code', $this->GATEWAY_CODE)->where('is_active', 1)->first();
    }

    public function getGatewaysModel(): ?Gateways
    {
        return $this->gateway;
    }

    public function setKey(): void
    {
        if (is_null($this->gateway)) {
            abort(404, 'Stripe gateway is not defined!');
        }

        $currency = Helper::findCurrencyFromId($this->gateway->id)->code;
        if ($this->gateway->mode === 'sandbox') {
            config(['cashier.key' => $this->gateway->sandbox_client_id]);
            config(['cashier.secret' => $this->gateway->sandbox_client_secret]);
            config(['cashier.currency' => $currency]);
            $key = $this->gateway->sandbox_client_secret;
        } else {
            config(['cashier.key' => $this->gateway->live_client_id]);
            config(['cashier.secret' => $this->gateway->live_client_secret]);
            config(['cashier.currency' => $currency]);
            $key = $this->gateway->live_client_secret;
        }
        Stripe::setApiKey($key);
        $this->key = $key;
    }

    public function setClient(): void
    {
        $this->stripe_client = new StripeClient($this->key);
    }

    public function createStripeCustomer($user): void
    {
        $userData = [
            'email'   => $user->email,
            'name'    => $user->name . ' ' . $user->surname,
            'phone'   => $user->phone,
            'address' => [
                'line1'       => $user->address,
                'postal_code' => $user->postal,
            ],
        ];
        $stripeId = $this->stripe_client->customers->create($userData)?->id;
        $user->stripe_id = $stripeId;
        $user->save();
    }

    public function checkoutData(User $user, ?int $planID): array
    {
        if ($user->stripe_id === null) {
            return [];
        }
        $plan = Plan::findOrFail($planID);

        return match ($plan->type) {
            TypeEnum::SUBSCRIPTION->value => $this->createSubscription($plan, $user),
            TypeEnum::TOKEN_PACK->value   => $this->createPrepaid($plan, $user),
            default                       => [],
        };
    }

    public function createSubscription(Plan $plan, User $user): array
    {
        DB::beginTransaction();

        try {
            $currency = $this->getCurrencyCode();
            $taxRate = $this->getTaxRate();
            $coupon = $this->checkCoupon();
            // $taxValue = $this->calculateTaxValue($plan->price, $taxRate);
            $taxRateId = $this->getOrCreateTaxRate($taxRate);

            $product = $this->getGatewayProduct($plan);
            if (! $product) {
                throw new RuntimeException('Product not found or missing price ID.');
            }

            $discountedPrice = $this->calculateDiscountedPrice($plan, $coupon);
            $discountedPriceCents = $this->convertToCents($discountedPrice);

            if ($this->isLifetimePlan($plan)) {
                $paymentIntent = $this->createLifetimeSubscription($plan, $user, $product, $discountedPriceCents, $currency);
            } else {
                $paymentIntent = $this->createRecurringSubscription($plan, $user, $product, $coupon, $taxRateId, $currency, $discountedPriceCents);
            }

            DB::commit();

            return [
                'paymentIntent'      => $paymentIntent,
            ];
        } catch (Exception $ex) {
            DB::rollBack();
            Log::error($this->GATEWAY_CODE . '-> createSubscription(): ' . $ex->getMessage());

            return [];
        }
    }

    public function createPrepaid(Plan $plan, User $user): array
    {
        DB::beginTransaction();

        try {
            $currency = $this->getCurrencyCode();
            $taxRate = $this->getTaxRate();
            $coupon = $this->checkCoupon();
            // $taxValue = $this->calculateTaxValue($plan->price, $taxRate);
            $taxRateId = $this->getOrCreateTaxRate($taxRate);

            $product = $this->getGatewayProduct($plan);
            if (! $product) {
                throw new RuntimeException('Product not found or missing price ID.');
            }

            $discountedPrice = $this->calculateDiscountedPrice($plan, $coupon);
            $discountedPriceCents = $this->convertToCents($discountedPrice);
            $paymentIntent = PaymentIntent::create([
                'amount'                    => $discountedPriceCents,
                'description'               => 'AI Services',
                'currency'                  => $currency,
                'customer'                  => $user->stripeId(),
                'automatic_payment_methods' => [
                    'enabled' => true,
                ],
                'metadata' => [
                    'product_id' => $product->product_id,
                    'price_id'   => $product->price_id,
                    'plan_id'    => $plan->id,
                ],
            ]);
            DB::commit();

            return [
                'paymentIntent'      => $paymentIntent,
            ];
        } catch (Exception $ex) {
            DB::rollBack();
            Log::error($this->GATEWAY_CODE . '-> createSubscription(): ' . $ex->getMessage());

            return [];
        }
    }

    public function subscribeCheckout(Request $request, $referral = null): \Illuminate\Http\RedirectResponse
    {
        $setup_intent = 'setup_intent';
        $payment_intent = 'payment_intent';
        $intentType = null;
        if ($request->has($payment_intent)) {
            $intentType = $payment_intent;
        } elseif ($request->has($setup_intent)) {
            $intentType = $setup_intent;
        }
        $intentId = $request->input($intentType);
        $clientSecret = $request->input($intentType . '_client_secret');
        $redirectStatus = $request->input('redirect_status');
        if ($redirectStatus !== 'succeeded') {
            return back()->with(['message' => __("A problem occurred! $redirectStatus"), 'type' => 'error']);
        }
        $intentStripe = null;
        if ($request->has($payment_intent)) {
            $intentStripe = 'paymentIntents';
        } elseif ($request->has($setup_intent)) {
            $intentStripe = 'setupIntents';
        }
        $intent = $this->stripe_client->{$intentStripe}->retrieve($intentId) ?? abort(404);
        if (! empty($intent->customer) && $intent->client_secret === $clientSecret && $intent->status === 'succeeded') {
            $customer = $this->stripe_client->customers->retrieve($intent->customer);
            $user = User::where('stripe_id', $customer->id)->where('email', $customer->email)->first();
            if ($user === null) {
                return back()->with(['message' => __('User not found!'), 'type' => 'error']);
            }

            $settings = Setting::getCache();
            $couponID = null;
            $previousRequest = app('request')->create(url()->previous());
            $subscription = Subscriptions::where('paid_with', $this->GATEWAY_CODE)->where(['user_id' => $user->id, 'stripe_status' => 'AwaitingPayment'])->latest()->first();

            try {
                DB::beginTransaction();
                // check validity of the intent
                $planId = $subscription->plan_id;
                $plan = Plan::where('id', $planId)->first();
                $total = $plan->price;
                $currency = Currency::where('id', $this->gateway->currency)->first()->code;
                $tax_rate_id = null;
                $taxValue = taxToVal($plan->price, $this->gateway->tax);
                // check the coupon existince
                if ($previousRequest->has('coupon')) {
                    $coupon = Coupon::where('code', $previousRequest->input('coupon'))->first();
                    if ($coupon) {
                        $coupon->usersUsed()->attach(auth()->user()->id);
                        $couponID = $coupon->discount;
                        $total -= ($plan->price * ($coupon->discount / 100));
                        if ($total != floor($total)) {
                            $total = number_format($total, 2);
                        }
                    }
                }
                $total += $taxValue;
                // update the subscription to make it active and save the total
                if ($subscription->auto_renewal) {
                    $subscription->stripe_status = 'stripe_approved';
                } else {
                    $subscription->stripe_status = $plan->trial_days != 0 ? 'trialing' : 'active';
                }
                $subscription->tax_rate = $this->gateway->tax;
                $subscription->tax_value = $taxValue;
                $subscription->coupon = $couponID;
                $subscription->total_amount = $total;
                $subscription->save();
                // save the order
                $order = new UserOrder;
                $order->order_id = $subscription->stripe_id;
                $order->plan_id = $planId;
                $order->user_id = $user->id;
                $order->payment_type = $this->GATEWAY_CODE;
                $order->price = $total;
                $order->affiliate_earnings = ($total * $settings->affiliate_commission_percentage) / 100;
                $order->status = 'Success';
                $order->country = Auth::user()->country ?? 'Unknown';
                $order->tax_rate = $this->gateway->tax;
                $order->tax_value = $taxValue;
                $order->save();
                \App\Models\Usage::getSingle()->updateSalesCount($total);
                DB::commit();
                $user->reg_sub_status = 'completed';
                $user->save();
                auth()->login($user);
                CreateActivity::for($user, __('Subscribed to'), $plan->name . ' ' . __('Plan'));
                self::creditIncreaseSubscribePlan($user, $plan);
                if (class_exists(AffiliateEvent::class)) {
                    event(new AffiliateEvent($total, $this->gateway->currency));
                }

                return redirect()->route('dashboard.user.payment.succesful')->with([
                    'message' => __('Thank you for your purchase. Enjoy your remaining words and images.'),
                    'type'    => 'success',
                ]);
            } catch (Exception $ex) {
                DB::rollBack();
                Log::error($this->GATEWAY_CODE . '-> subscribeCheckout(): ' . $ex->getMessage());

                return back()->with(['message' => Str::before($ex->getMessage(), ':'), 'type' => 'error']);
            }
        }

        return back()->with(['message' => __('A problem occurred!'), 'type' => 'error']);
    }

    public function prepaidCheckout(Request $request, $referral = null)
    {
        $gateway = $this->gateway;
        $settings = Setting::getCache();
        $stripe = $this->stripe_client;
        $previousRequest = app('request')->create(url()->previous());
        if ($request->has('payment_intent') && $request->has('payment_intent_client_secret') && $request->has('redirect_status')) {
            $payment_intent = $request->input('payment_intent');
            $payment_intent_client_secret = $request->input('payment_intent_client_secret');
            $redirect_status = $request->input('redirect_status');
            if ($redirect_status !== 'succeeded') {
                return back()->with(['message' => __("A problem occurred! $redirect_status"), 'type' => 'error']);
            }
            $intent = $stripe->paymentIntents->retrieve($payment_intent);
            if (empty($intent->customer) || $intent == null || $intent->client_secret != $payment_intent_client_secret || $intent->status != 'succeeded') {
                return back()->with(['message' => __('A problem occurred!'), 'type' => 'error']);
            }
            $customer = $this->stripe_client->customers->retrieve($intent->customer);
            $user = User::where('stripe_id', $customer->id)->where('email', $customer->email)->first();
            if ($user === null) {
                return back()->with(['message' => __('User not found!'), 'type' => 'error']);
            }

            try {
                DB::beginTransaction();

                $planId = $intent->metadata->plan_id;
                $productId = $intent->metadata->product_id;
                $priceId = $intent->metadata->price_id;
                $plan = Plan::where('id', $planId)->first();
                if ($plan == null) {
                    return back()->with(['message' => __('A problem occurred!'), 'type' => 'error']);
                }

                $total = $plan->price;
                if ($previousRequest->has('coupon')) {
                    $coupon = Coupon::where('code', $previousRequest->input('coupon'))->first();
                    if ($coupon) {
                        $couponID = $coupon->discount;
                        $total -= ($plan->price * ($coupon->discount / 100));
                        if ($total != floor($total)) {
                            $total = number_format($total, 2);
                        }
                        $coupon->usersUsed()->attach(auth()->user()->id);
                    }
                }
                $total += taxToVal($plan->price, $gateway->tax);

                $order = new UserOrder;
                $order->order_id = 'SPO-' . strtoupper(Str::random(13));
                $order->plan_id = $plan->id;
                $order->user_id = $user->id;
                $order->type = 'prepaid';
                $order->payment_type = $this->GATEWAY_CODE;
                $order->price = $total;
                $order->affiliate_earnings = ($total * $settings->affiliate_commission_percentage) / 100;
                $order->status = 'Success';
                $order->country = $user->country ?? 'Unknown';
                $order->tax_rate = $gateway->tax;
                $order->tax_value = taxToVal($plan->price, $gateway->tax);
                $order->save();
                auth()->login($user);
                self::creditIncreaseSubscribePlan($user, $plan);
                // check if any other "AwaitingPayment" subscription exists if so cancel it
                $waiting_subscriptions = Subscriptions::where('paid_with', $this->GATEWAY_CODE)->where(['user_id' => $user->id, 'stripe_status' => 'AwaitingPayment'])->get();
                foreach ($waiting_subscriptions as $waitingSubs) {
                    dispatch(new CancelAwaitingPaymentSubscriptions($stripe, $waitingSubs));
                }
                CreateActivity::for($user, __('Purchased'), $plan->name . ' ' . __('Plan'));
                \App\Models\Usage::getSingle()->updateSalesCount($total);
            } catch (Exception $th) {
                DB::rollBack();
                Log::error($this->GATEWAY_CODE . '-> prepaidCheckout(): ' . $th->getMessage());

                return back()->with(['message' => Str::before($th->getMessage(), ':'), 'type' => 'error']);
            }
        } else {
            return back()->with(['message' => __('A problem occurred!'), 'type' => 'error']);
        }
        DB::commit();

        return redirect()->route('dashboard.user.payment.succesful')->with([
            'message' => __('Thank you for your purchase. Enjoy your remaining words and images.'),
            'type'    => 'success',
        ]);

    }

    private function getCurrencyCode(): string
    {
        return Helper::findCurrencyFromId($this->gateway?->id)->code ?? 'USD';
    }

    private function getTaxRate(): float
    {
        return (float) $this->gateway?->tax;
    }

    private function checkCoupon(): ?Coupon
    {
        return checkCouponInRequest();
    }

    private function calculateTaxValue(float $price, float $taxRate): float
    {
        return taxToVal($price, $taxRate);
    }

    private function getOrCreateTaxRate(float $taxRate): ?string
    {
        if ($taxRate <= 0) {
            return null;
        }

        $stripeTaxRates = $this->stripe_client?->taxRates->all()->data ?? [];
        foreach ($stripeTaxRates as $sTax) {
            if ($sTax->percentage === $taxRate) {
                return $sTax->id;
            }
        }

        $newTax = $this->stripe_client?->taxRates->create([
            'percentage'   => $taxRate,
            'display_name' => Str::random(13),
            'inclusive'    => false,
        ]);

        return $newTax->id ?? null;
    }

    private function getGatewayProduct(Plan $plan)
    {
        return GatewayProducts::where(['plan_id' => $plan->id, 'gateway_code' => $this->GATEWAY_CODE])->first();
    }

    private function calculateDiscountedPrice(Plan $plan, $coupon): float
    {
        if (! $coupon) {
            return $plan->price;
        }

        $discountedPrice = $plan->price - ($plan->price * ($coupon->discount / 100));

        return round($discountedPrice, 2);
    }

    private function convertToCents(float $amount): int
    {
        return (int) ($amount * 100);
    }

    private function isLifetimePlan(Plan $plan): bool
    {
        return in_array($plan->frequency, [FrequencyEnum::LIFETIME_MONTHLY->value, FrequencyEnum::LIFETIME_YEARLY->value], true);
    }

    private function createLifetimeSubscription(Plan $plan, User $user, $product, int $amountCents, string $currency): PaymentIntent
    {
        $subscription = new Subscriptions([
            'user_id'       => $user->id,
            'name'          => $plan->id,
            'stripe_id'     => 'SLS-' . strtoupper(Str::random(13)),
            'stripe_status' => 'AwaitingPayment',
            'stripe_price'  => $product->price_id,
            'quantity'      => 1,
            'trial_ends_at' => null,
            'ends_at'       => $plan->frequency === FrequencyEnum::LIFETIME_MONTHLY->value ? Carbon::now()->addMonths(1) : Carbon::now()->addYears(1),
            'auto_renewal'  => 1,
            'plan_id'       => $plan->id,
            'paid_with'     => $this->GATEWAY_CODE,
        ]);
        $subscription->save();

        return PaymentIntent::create([
            'amount'                    => $amountCents,
            'currency'                  => $currency,
            'customer'                  => $user->stripeId(),
            'description'               => 'AI Services',
            'automatic_payment_methods' => ['enabled' => true],
            'metadata'                  => [
                'product_id' => $product->product_id,
                'price_id'   => $product->price_id,
                'plan_id'    => $plan->id,
                'type'       => 'prepaid',
            ],
        ]);
    }

    /**
     * @throws ApiErrorException
     */
    private function createRecurringSubscription(Plan $plan, User $user, $product, $coupon, $taxRateId, string $currency, int $discountedPriceCents): array
    {
        $subscriptionInfo = [
            'customer'         => $user->stripe_id,
            'items'            => [['price' => $product->price_id, 'tax_rates' => $taxRateId ? [$taxRateId] : []]],
            'payment_behavior' => 'default_incomplete',
            'payment_settings' => ['save_default_payment_method' => 'on_subscription'],
            'expand'           => ['latest_invoice.payment_intent'],
            'metadata'         => [
                'product_id' => $product->product_id,
                'price_id'   => $product->price_id,
                'plan_id'    => $plan->id,
            ],
        ];

        if ($coupon) {
            $subscriptionInfo['coupon'] = $this->getOrCreateCoupon($coupon);
        }

        if ($plan->trial_days !== 0) {
            $trialEndTimestamp = Carbon::now()->addDays($plan->trial_days)->timestamp;
            $subscriptionInfo += [
                'trial_end'            => $trialEndTimestamp,
                'billing_cycle_anchor' => $trialEndTimestamp,
            ];
        }

        $newSubscription = $this->stripe_client?->subscriptions->create($subscriptionInfo);

        $subscription = new Subscriptions([
            'user_id'       => $user->id,
            'name'          => $plan->id,
            'stripe_id'     => $newSubscription->id,
            'stripe_status' => 'AwaitingPayment',
            'stripe_price'  => $product->price_id,
            'quantity'      => 1,
            'trial_ends_at' => $plan->trial_days !== 0 ? Carbon::now()->addDays($plan->trial_days) : null,
            'ends_at'       => $plan->trial_days !== 0 ? Carbon::now()->addDays($plan->trial_days) : Carbon::now()->addDays(30),
            'plan_id'       => $plan->id,
            'paid_with'     => $this->GATEWAY_CODE,
        ]);
        $subscription->save();

        return [
            'subscription_id' => $newSubscription->id,
            'client_secret'   => $plan->trial_days !== 0
                ? $this->stripe_client?->setupIntents->retrieve($newSubscription->pending_setup_intent, [])->client_secret
                : $newSubscription->latest_invoice->payment_intent->client_secret,
            'trial'       => $plan->trial_days !== 0,
            'currency'    => $currency,
            'amount'      => $discountedPriceCents,
            'description' => 'AI Services',
            'type'        => 'subscription',
        ];
    }

    private function getOrCreateCoupon($coupon): ?string
    {
        $stripeCoupons = $this->stripe_client?->coupons->all()->data ?? [];
        foreach ($stripeCoupons as $sCoupon) {
            if ($sCoupon->percent_off === $coupon->discount) {
                return $sCoupon->id;
            }
        }

        $newCoupon = $this->stripe_client?->coupons->create([
            'percent_off' => $coupon->discount,
            'duration'    => 'once',
        ]);

        return $newCoupon->id ?? null;
    }
}
