<?php

namespace App\Services\Payment\Gateways;

use App\Actions\CreateActivity;
use App\Actions\EmailPaymentConfirmation;
use App\Enums\Plan\FrequencyEnum;
use App\Enums\Plan\TypeEnum;
use App\Helpers\Classes\Helper;
use App\Jobs\ProcessGatewayCustomerJob;
use App\Models\GatewayProducts;
use App\Models\Plan;
use App\Models\Setting;
use App\Models\User;
use App\Models\UserOrder;
use App\Services\Payment\Contracts\AbstractPaymentGateway;
use App\Services\Payment\Contracts\PaymentGatewayInterface;
use App\Services\Payment\Enums\PaymentGatewayEnum;
use App\Services\PaymentGateways\Contracts\CreditUpdater;
use Exception;
use Illuminate\Contracts\Auth\Authenticatable;
use Illuminate\Contracts\View\View;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Config;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;
use Laravel\Cashier\Subscription as Subscriptions;
use RuntimeException;

class TwoCheckoutGateway extends AbstractPaymentGateway implements PaymentGatewayInterface
{
    use CreditUpdater;

    private const BASE_URL = 'https://api.2checkout.com/rest/6.0';

    private const CUSTOMERS_ENDPOINT = '/customers';

    private const PRODUCTS_ENDPOINT = '/products';

    private const ORDERS_ENDPOINT = '/orders';

    private ?string $secretKey;

    private ?string $merchantCode;

    public static function enum(): PaymentGatewayEnum
    {
        return PaymentGatewayEnum::TwoCheckout;
    }

    public function createCustomer(Authenticatable|User|null $user): void
    {
        if (is_null($user)) {
            throw new RuntimeException('Invalid user provided for customer creation.');
        }

        try {
            $payload = array_filter([
                'ExternalCustomerReference' => null,
                'FirstName'                 => $user->name,
                'LastName'                  => $user->surname,
                'Company'                   => $user->company_name,
                'FiscalCode'                => null,
                'Address1'                  => $user->address ?? 'sample address',
                'Address2'                  => null,
                'City'                      => $user->city ?? 'sample city',
                'State'                     => $user->state ?? 'sample state',
                'Zip'                       => $user->postal ?? '12345',
                'CountryCode'               => 'US',
                'Phone'                     => $user->phone,
                'Fax'                       => null,
                'Email'                     => $user->email,
                'ExistingCards'             => [],
                'Enabled'                   => true,
                'Trial'                     => false,
                'Language'                  => 'EN',
                'CustomerReference'         => null,
                'Credit'                    => null,
            ], static fn ($value) => ! is_null($value));
            $customerReference = $this->sendRequest('POST', self::CUSTOMERS_ENDPOINT, $payload);
            if (! empty($customerReference)) {
                $user->update(['two_checkout_customer_reference' => $customerReference]);
            }
        } catch (Exception $ex) {
            Log::error(self::enum()->label() . '-> createCustomer(): ' . $ex->getMessage());
        }
    }

    public function saveProduct(Plan $plan): void
    {
        set_time_limit(0);
        ini_set('max_execution_time', 9000);

        try {
            $gateway = self::getGateway();
            DB::beginTransaction();
            $currency = Helper::findCurrencyFromId($gateway?->currency)?->getAttribute('code');
            $pCode = strtoupper(Str::random(10));
            $data = [
                'ProductCode'           => $pCode,
                'ProductName'           => $plan->name,
                'Enabled'               => true,
                'PricingConfigurations' => [
                    [
                        'Name'            => $plan->name . '\'s Price Configuration',
                        'Code'            => Str::random(10),
                        'Default'         => true,
                        'PricingSchema'   => 'DYNAMIC',
                        'PriceType'       => 'NET',
                        'DefaultCurrency' => $currency,
                        'Prices'          => [
                            'Regular' => [
                                [
                                    'Amount'      => $plan->price,
                                    'Currency'    => $currency,
                                    'MinQuantity' => 1,
                                    'MaxQuantity' => 99999,
                                ],
                            ],
                            'Renewal' => [],
                        ],
                    ],
                ],
            ];

            $isSubscriptionPlan = $plan->price !== 0 && $plan->type === TypeEnum::SUBSCRIPTION->value && ! in_array($plan->frequency, [
                FrequencyEnum::LIFETIME_MONTHLY->value,
                FrequencyEnum::LIFETIME_YEARLY->value,
            ], true);

            if ($isSubscriptionPlan) {
                $data['GeneratesSubscription'] = true;
                $data['SubscriptionInformation'] = [
                    'BillingCycle'      => $plan->frequency === FrequencyEnum::MONTHLY->value ? 1 : 12,
                    'BillingCycleUnits' => 'M',
                    'IsOneTimeFee'      => false,
                ];
            }
            $newProduct = $this->sendRequest('post', self::PRODUCTS_ENDPOINT, $data);
            if (! empty($newProduct)) {
                $product = new GatewayProducts;
                $product->plan_id = $plan->id;
                $product->gateway_code = PaymentGatewayEnum::TwoCheckout->value;
                $product->gateway_title = PaymentGatewayEnum::TwoCheckout->label();
                $product->product_id = $pCode;
                $product->plan_name = $plan->name;

                if ($isSubscriptionPlan) {
                    $createdProducts = $this->getProducts();
                    $foundProduct = $createdProducts?->firstWhere('ProductCode', $pCode);
                    if ($foundProduct) {
                        $product->price_id = $foundProduct['PricingConfigurations'][0]['Code'] ?? null;
                    }
                } else {
                    $product->price_id = 'Not Needed';
                }
                $product->save();
                DB::commit();
            }
            DB::rollBack();
        } catch (Exception $ex) {
            Log::error(self::enum()->label() . '-> saveProduct(): ' . $ex->getMessage());
            DB::rollBack();
        }
    }

    public function saveAllProducts(): void
    {
        try {
            $customerWithoutRef = User::whereNull('two_checkout_customer_reference')->get();
            foreach ($customerWithoutRef as $user) {
                dispatch(new ProcessGatewayCustomerJob(PaymentGatewayEnum::TwoCheckout, $user));
            }

            $plans = Plan::query()->where('active', 1)->get();

            foreach ($plans as $plan) {
                $this->saveProduct($plan);
            }

            // $this->createWebhook(); create a webhook if needed
        } catch (Exception $ex) {
            Log::error(self::enum()->label() . '-> saveAllProducts(): ' . $ex->getMessage());
        }
    }

    public function subscribe(Plan $plan): View|RedirectResponse
    {
        try {
            DB::beginTransaction();
            $gateway = self::getGateway();
            $user = auth()->user();

            $this->createCustomerIfNotExist($user);

            $product = GatewayProducts::where(['plan_id' => $plan->id, 'gateway_code' => PaymentGatewayEnum::TwoCheckout->value])->first();
            $this->checkProductAndPriceId($product);

            $price_id_product = $product->price_id;
            $newDiscountedPrice = $plan->price;

            $subscription = new Subscriptions;
            $subscription->user_id = $user?->id;
            $subscription->name = $plan->id;
            $subscription->stripe_status = 'AwaitingPayment';
            $subscription->quantity = 1;
            $subscription->stripe_price = $price_id_product;
            $subscription->plan_id = $plan->id;
            $subscription->paid_with = self::enum()->value;
            $subscription->total_amount = $newDiscountedPrice;

            if ($plan->frequency === FrequencyEnum::LIFETIME_MONTHLY->value || $plan->frequency === FrequencyEnum::LIFETIME_YEARLY->value) {
                $subscription->stripe_id = 'TLS-' . strtoupper(Str::random(13));
                $subscription->trial_ends_at = null;
                $subscription->ends_at = $plan->frequency === FrequencyEnum::LIFETIME_MONTHLY->value ? Carbon::now()->addMonths(1) : Carbon::now()->addYears(1);
                $subscription->auto_renewal = 1;
            } else {
                $subscription->stripe_id = 'TS-' . strtoupper(Str::random(13));
                $subscription->trial_ends_at = $plan->trial_days != 0 ? Carbon::now()->addDays($plan->trial_days) : null;
                $subscription->ends_at = $plan->trial_days != 0 ? Carbon::now()->addDays($plan->trial_days) : Carbon::now()->addDays(30);
            }
            $subscription->save();
            $mCode = $this->merchantCode;
            DB::commit();

            return view('panel.user.finance.subscription.' . self::enum()->value, compact('plan', 'mCode', 'newDiscountedPrice', 'price_id_product', 'gateway', 'product'));
        } catch (Exception $ex) {
            DB::rollBack();
            Log::error(self::enum()->label() . '-> subscribe(): ' . $ex->getMessage());

            return back()->with(['message' => Str::before($ex->getMessage(), ':'), 'type' => 'error']);
        }
    }

    public function prepaid(Plan $plan): View|RedirectResponse
    {
        try {
            $gateway = self::getGateway();
            $user = auth()->user();

            $this->createCustomerIfNotExist($user);
            $product = GatewayProducts::where(['plan_id' => $plan->id, 'gateway_code' => PaymentGatewayEnum::TwoCheckout->value])->first();
            $this->checkProductAndPriceId($product);
            $newDiscountedPrice = $plan->price;
            $mCode = $this->merchantCode;

            return view('panel.user.finance.prepaid.' . self::enum()->value, compact('plan', 'newDiscountedPrice', 'gateway', 'mCode', 'product'));
        } catch (Exception $th) {
            Log::error(self::enum()->label() . '-> prepaid(): ' . $th->getMessage());

            return back()->with(['message' => Str::before($th->getMessage(), ':'), 'type' => 'error']);
        }
    }

    public function subscribeCheckout(Request $request, $referral = null): RedirectResponse
    {
        $settings = Setting::getCache();
        $refNo = $request->get('refno');
        // validate signature later https://verifone.cloud/docs/2checkout/Documentation/07Commerce/InLine-Checkout-Guide/Signature_validation_for_return_URL_via_InLine_checkout
        $signature = $request->get('signature');

        if (! $refNo || ! $signature) {
            return redirect()->back()->with(['message' => __('Invalid request parameters.'), 'type' => 'error']);
        }

        try {
            DB::beginTransaction();
            $order = $this->getOrder($refNo);
            if (empty($order)) {
                return redirect()->back()->with(['message' => __('Order not found.'), 'type' => 'error']);
            }

            $user = auth()->user();
            $status = $order['Status'] ?? 'PENDING';
            $approveStatus = $order['ApprovalStatus'] ?? 'WAITING';
            if ($status === 'COMPLETE' && $approveStatus === 'OK') {
                $item = $order['Items'][0] ?? null;
                if (! $item) {
                    DB::rollBack();

                    return redirect()->back()->with(['message' => __('Order item not found.'), 'type' => 'error']);
                }

                $product = GatewayProducts::where(['product_id' => $item['ProductCode'], 'gateway_code' => PaymentGatewayEnum::TwoCheckout->value])->first();
                if (! $product) {
                    DB::rollBack();

                    return redirect()->back()->with(['message' => __('Product not found.'), 'type' => 'error']);
                }

                $plan = Plan::find($product->plan_id);
                if (! $plan) {
                    DB::rollBack();

                    return redirect()->back()->with(['message' => __('Plan not found.'), 'type' => 'error']);
                }

                $subscription = Subscriptions::query()
                    ->where('user_id', $user?->id)
                    ->where('plan_id', $plan->id)
                    ->where('stripe_status', 'AwaitingPayment')
                    ->where('paid_with', self::enum()->value)
                    ->first();

                if (! $subscription) {
                    DB::rollBack();

                    return redirect()->back()->with(['message' => __('Subscription not found.'), 'type' => 'error']);
                }

                $subscription->stripe_status = 'active';
                $subscription->save();

                // save the order
                $order = new UserOrder;
                $order->order_id = $subscription->stripe_id;
                $order->plan_id = $plan->id;
                $order->user_id = $user?->id;
                $order->payment_type = self::enum()->value;
                $order->price = $plan->price;
                $order->affiliate_earnings = ($plan->price * $settings->affiliate_commission_percentage) / 100;
                $order->status = 'Success';
                $order->country = Auth::user()->country ?? 'Unknown';
                $order->save();

                self::creditIncreaseSubscribePlan($user, $plan);

                CreateActivity::for($user, __('Subscribed to'), $plan->name . ' ' . __('Plan'));
                EmailPaymentConfirmation::create($user, $plan)->send();
                \App\Models\Usage::getSingle()->updateSalesCount($plan->price);

                DB::commit();

                return redirect()->route('dashboard.user.payment.succesful')->with([
                    'message' => __('Thank you for your purchase. Enjoy your remaining words and images.'),
                    'type'    => 'success',
                ]);
            }

            DB::rollBack();

            return redirect()->back()->with([
                'message' => __('Your payment is still pending or was not approved. Please try again later.'),
                'type'    => 'warning',
            ]);
        } catch (Exception $ex) {
            Log::error(self::enum()->label() . '-> subscribeCheckout(): ' . $ex->getMessage());
            DB::rollBack();

            return redirect()->back()->with(['message' => __('Error fetching order details.'), 'type' => 'error']);
        }
    }

    public function prepaidCheckout(Request $request, $referral = null): RedirectResponse
    {
        $settings = Setting::getCache();
        $refNo = $request->get('refno');
        // validate signature later https://verifone.cloud/docs/2checkout/Documentation/07Commerce/InLine-Checkout-Guide/Signature_validation_for_return_URL_via_InLine_checkout
        $signature = $request->get('signature');

        if (! $refNo || ! $signature) {
            return redirect()->back()->with(['message' => __('Invalid request parameters.'), 'type' => 'error']);
        }

        try {
            DB::beginTransaction();
            $order = $this->getOrder($refNo);
            if (empty($order)) {
                return redirect()->back()->with(['message' => __('Order not found.'), 'type' => 'error']);
            }

            $user = auth()->user();
            $status = $order['Status'] ?? 'PENDING';
            $approveStatus = $order['ApprovalStatus'] ?? 'WAITING';
            if ($status === 'COMPLETE' && $approveStatus === 'OK') {
                $item = $order['Items'][0] ?? null;
                if (! $item) {
                    DB::rollBack();

                    return redirect()->back()->with(['message' => __('Order item not found.'), 'type' => 'error']);
                }

                $product = GatewayProducts::where(['product_id' => $item['ProductCode'], 'gateway_code' => PaymentGatewayEnum::TwoCheckout->value])->first();
                if (! $product) {
                    DB::rollBack();

                    return redirect()->back()->with(['message' => __('Product not found.'), 'type' => 'error']);
                }

                $plan = Plan::find($product->plan_id);
                if (! $plan) {
                    DB::rollBack();

                    return redirect()->back()->with(['message' => __('Plan not found.'), 'type' => 'error']);
                }
                $order = new UserOrder;
                $order->order_id = 'TPO-' . strtoupper(Str::random(13));
                $order->plan_id = $plan->id;
                $order->user_id = $user?->id;
                $order->type = 'prepaid';
                $order->payment_type = self::enum()->value;
                $order->price = $plan->price;
                $order->affiliate_earnings = ($plan->price * $settings->affiliate_commission_percentage) / 100;
                $order->status = 'Success';
                $order->country = $user->country ?? 'Unknown';
                $order->save();

                self::creditIncreaseSubscribePlan($user, $plan);
                CreateActivity::for($user, __('Purchased'), $plan->name . ' ' . __('Plan'));
                EmailPaymentConfirmation::create($user, $plan)->send();
                \App\Models\Usage::getSingle()->updateSalesCount($plan->price);

                DB::commit();

                return redirect()->route('dashboard.user.payment.succesful')->with([
                    'message' => __('Thank you for your purchase. Enjoy your remaining words and images.'),
                    'type'    => 'success',
                ]);
            }

            DB::rollBack();

            return redirect()->back()->with([
                'message' => __('Your payment is still pending or was not approved. Please try again later.'),
                'type'    => 'warning',
            ]);
        } catch (Exception $ex) {
            Log::error(self::enum()->label() . '-> subscribeCheckout(): ' . $ex->getMessage());
            DB::rollBack();

            return redirect()->back()->with(['message' => __('Error fetching order details.'), 'type' => 'error']);
        }
    }

    public function getSubscriptionStatus($incomingUserId = null): bool
    {
        $incomingUserId === null ? $user = Auth::user() : $user = User::where('id', $incomingUserId)->first();
        $sub = getCurrentActiveSubscription($user->id);

        return $sub && $sub->stripe_status === 'active';
    }

    public function getSubscriptionDaysLeft(): int
    {
        $activeSub = getCurrentActiveSubscription(Auth::id());
        if ($activeSub->status === 'active') {
            return Carbon::now()->diffInDays(Carbon::createFromTimeStamp($activeSub->ends_at));
        }

        return 0;
    }

    public function getSubscriptionRenewDate(): int|string|null
    {
        $activeSub = getCurrentActiveSubscription(Auth::id());
        $end = $activeSub->ends_at;

        return Carbon::createFromTimeStamp($end)->format('F jS, Y');
    }

    public function checkIfTrial(): bool
    {
        return false;
    }

    public function subscribeCancel(null|Authenticatable|User $user, ?string $msg): RedirectResponse
    {
        // cancel subscription from 2checkout also
        $activeSub = getCurrentActiveSubscription($user->id);
        if ($activeSub && $activeSub->stripe_status === 'active') {
            $activeSub->stripe_status = 'cancelled';
            $activeSub->save();
            $plan = Plan::where('id', $activeSub->plan_id)->first();
            self::creditDecreaseCancelPlan($user, $plan);
            CreateActivity::for($user, __('Cancelled'), $plan->name . ' ' . __('Plan'));

            return redirect()->route('dashboard.user.index')->with(['message' => $msg ?? __('Your subscription is cancelled successfully.'), 'type' => 'success']);
        }

        return redirect()->route('dashboard.user.index')->with(['message' => __('No active subscription found.'), 'type' => 'error']);
    }

    public function handleWebhook(Request $request): JsonResponse
    {
        return response()->json();
    }

    public function cancelSubscribedPlan(): bool
    {
        $tmp = $this->subscribeCancel(Auth::user(), __('Subscription is cancelled successfully.'));

        return true;
    }

    public function getPlansPriceIdsForMigration(Subscriptions $subscription, ?int $planId): void {}

    public function getUsersCustomerIdsForMigration(Subscriptions $subscription): void {}

    // helper functions below

    protected function checkProductAndPriceId(?GatewayProducts $product): ?RedirectResponse
    {
        if (! $product) {
            $exception = __('Product is not defined! Please save Membership Plan again.');

            return back()->with(['message' => $exception, 'type' => 'error']);
        }
        if (! $product->price_id) {
            $exception = __('Product ID is not set! Please save Membership Plan again.');

            return back()->with(['message' => $exception, 'type' => 'error']);
        }

        return null;
    }

    protected function createCustomerIfNotExist(Authenticatable|User|null $user): void
    {
        try {
            if (! empty($user?->two_checkout_customer_reference)) {
                // Check if the customer already exists in 2checkout
                $this->getCustomer($user?->two_checkout_customer_reference);
            } else {
                // Customer doesn't exist, create a new customer
                $this->createCustomer($user);
            }
        } catch (\Stripe\Exception\InvalidRequestException $e) {
            // Customer doesn't exist, create a new customer
            $this->createCustomer($user);
        }
    }

    protected function loadConfig(): void
    {
        $this->merchantCode = config('2checkout.merchant_code');
        $this->secretKey = config('2checkout.secret_key');

        if (! $this->merchantCode || ! $this->secretKey) {
            $gateway = self::getGateway();
            $this->merchantCode = $gateway?->getAttribute('live_app_id');
            $this->secretKey = $gateway?->getAttribute('live_client_secret');

            Config::set('2checkout.merchant_code', $this->merchantCode);
            Config::set('2checkout.secret_key', $this->secretKey);
        }
    }

    protected function buildAuthHeader(): string
    {
        $date = gmdate('Y-m-d H:i:s');
        $stringToHash = strlen($this->merchantCode) . $this->merchantCode . strlen($date) . $date;
        $hash = hash_hmac('md5', $stringToHash, $this->secretKey);

        return 'code="' . $this->merchantCode . '" date="' . $date . '" hash="' . $hash . '"';
    }

    public function sendRequest(string $method, string $endpoint, array $payload = []): array|int|string
    {
        $url = rtrim(self::BASE_URL, '/') . '/' . ltrim($endpoint, '/');
        $response = Http::withHeaders([
            'Accept'                    => 'application/json',
            'X-Avangate-Authentication' => $this->buildAuthHeader(),
        ])->$method($url, $payload);

        if (! $response->successful()) {
            throw new RuntimeException("2Checkout API Error: {$response->status()} - {$response->body()}");
        }

        return $response->json();
    }

    protected function getCustomer(string $customerReference): array
    {
        return $this->sendRequest('GET', self::CUSTOMERS_ENDPOINT . '/' . $customerReference);
    }

    protected function getProducts(): Collection
    {
        try {
            $limit = [
                'Limit' => 100,
            ];
            $products = $this->sendRequest('get', self::PRODUCTS_ENDPOINT, $limit);

            return collect($products['Items']);
        } catch (Exception $ex) {
            Log::error(self::enum()->label() . '-> getProducts(): ' . $ex->getMessage());

            return collect();
        }
    }

    protected function getOrder(string $orderReference): array
    {
        return $this->sendRequest('GET', self::ORDERS_ENDPOINT . '/' . $orderReference);
    }
}
