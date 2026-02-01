<?php

namespace App\Extensions\CheckoutRegistration\System\Http\Controllers;

use App\Actions\EmailConfirmation;
use App\Extensions\CheckoutRegistration\System\Http\Services\Finance\PaypalService;
use App\Extensions\CheckoutRegistration\System\Http\Services\Finance\StripeService;
use App\Helpers\Classes\MarketplaceHelper;
use App\Http\Controllers\Controller;
use App\Models\Plan;
use App\Models\Team\TeamMember;
use App\Models\User;
use Exception;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Str;

class RegisterCheckoutController extends Controller
{
    protected Exception|StripeService|PaypalService $gatewayService;

    /**
     * @throws Exception
     */
    public function __construct()
    {
        $this->gatewayService = $this->chooseService();
    }

    public function index(Request $request)
    {
        $plan = $this->getPlan($request);
        $gatewayService = $this->gatewayService;

        $view = 'checkout-registration::auth.register';
        if ((setting('checkout_registration_status', 'passive') === 'passive') || ! MarketplaceHelper::isRegistered('checkout-registration')) {
            $plan = $this->getPlan($request)?->id;
            $view = 'panel.authentication.register';
        }

        return view($view, compact('plan', 'gatewayService'));
    }

    public function store(Request $request): JsonResponse
    {
        $validator = validator($request->all(), [
            'email'   => [
                'required',
                'string',
                'email',
                'max:255',
                function ($attribute, $value, $fail) {
                    $existingUser = User::where('email', $value)
                        ->where('reg_sub_status', 'completed')
                        ->exists();

                    if ($existingUser) {
                        $fail('The email is already associated with a completed registration.');
                    }
                },
            ],
            'planID'  => ['required', 'integer'],
        ]);
        if ($validator->fails()) {
            return response()->json(['message' => $validator->errors()->first()], 422);
        }
        $inProgressUser = User::where('email', $request->email)
            ->where('reg_sub_status', 'in_progress');
        if (! $inProgressUser->exists()) {
            $teamMember = TeamMember::query()
                ->with('team')
                ->where('email', $request->email)
                ->where('status', 'waiting')
                ->first();
            $affCode = null;
            if ($request->affiliate_code !== null) {
                $affUser = User::where('affiliate_code', $request->affiliate_code)->first();
                $affCode = $affUser?->id;
            }
            $userName = explode('@', $request->email);
            $user = User::create([
                'team_id'                 => $teamMember?->team_id,
                'team_manager_id'         => $teamMember?->team?->user_id,
                'name'                    => $userName[0],
                'surname'                 => '',
                'status'                  => 0,
                'reg_sub_status'          => 'in_progress',
                'email'                   => $request->email,
                'email_confirmation_code' => Str::random(67),
                'password'                => Hash::make(Str::random(12)),
                'email_verification_code' => Str::random(67),
                'affiliate_id'            => $affCode,
                'affiliate_code'          => Str::upper(Str::random(12)),
            ]);
            $user->updateCredits(setting('freeCreditsUponRegistration', User::getFreshCredits()));
            $teamMember?->update([
                'user_id'   => $user->id,
                'status'    => 'active',
                'joined_at' => now(),
            ]);
            EmailConfirmation::forUser($user)->send();
        } else {
            $user = $inProgressUser->first();
        }

        $this->gatewayService->createPreReqsIfNeeded($user);
        $checkoutData = $this->gatewayService->checkoutData($user, $request->planID);

        return response()->json(['message' => 'success', 'checkoutData' => $checkoutData]);
    }

    public function checkout(Request $request, $referral = null)
    {
        if ($request->type === 'subscribe') {
            return $this->gatewayService->subscribeCheckout($request, $referral);
        }

        return $this->gatewayService->prepaidCheckout($request, $referral);
    }

    /**
     * @throws Exception
     */
    private function chooseService(): StripeService|PaypalService|Exception
    {
        return match (setting('default_checkout_gateway', 'stripe')) {
            'stripe' => new StripeService,
            'paypal' => new PaypalService,
            default  => throw new Exception('Gateway not found'),
        };
    }

    private function getPlan(Request $request)
    {
        $request->validate([
            'plan' => 'sometimes|nullable',
        ]);
        $plan_id = $request->get('plan') ?? setting('default_checkout_plan_id', Plan::where('active', 1)->first()?->id);
        if ($plan_id) {
            return Plan::find($plan_id) ?? Plan::where('active', 1)->first();
        }

        return Plan::where('active', 1)->first();
    }
}
