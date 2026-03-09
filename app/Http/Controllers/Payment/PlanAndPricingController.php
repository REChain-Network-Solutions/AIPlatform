<?php

namespace App\Http\Controllers\Payment;

use App\Domains\Entity\EntityStats;
use App\Enums\Plan\FrequencyEnum;
use App\Enums\Plan\TypeEnum;
use App\Http\Controllers\Controller;
use App\Models\Gateways;
use App\Models\OpenAIGenerator;
use App\Models\Plan;
use App\Models\Team\Team;
use App\Models\User;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Cache;

class PlanAndPricingController extends Controller
{
    public function __invoke()
    {
        $activeGateways = Gateways::query()->where('is_active', 1)->get();

        $is_active_gateway = $activeGateways->count() > 0 ? 1 : 0;

        // Check if any subscription is active
        $userId = Auth::id();
        $activeSub = getCurrentActiveSubscription($userId) ?? getCurrentActiveSubscriptionYokkasa($userId);
        $activesubid = $activeSub ? $activeSub->plan_id : 0;

        $plans = Plan::query()
            ->where('active', 1)
            ->where('hidden', false)
            ->get();

        $plansSubscriptionMonthly = $plans->where('type', TypeEnum::SUBSCRIPTION->value)
            ->where('frequency', FrequencyEnum::MONTHLY->value)
            ->sortBy('price');

        $plansSubscriptionLifetime = $plans->where('type', TypeEnum::SUBSCRIPTION->value)
            ->whereIn('frequency', [FrequencyEnum::LIFETIME_YEARLY->value, FrequencyEnum::LIFETIME_MONTHLY->value])
            ->sortBy('price');

        $plansSubscriptionAnnual = $plans->where('type', TypeEnum::SUBSCRIPTION->value)
            ->where('frequency', FrequencyEnum::YEARLY->value)
            ->sortBy('price');

        $prepaidplans = $plans->where('type', TypeEnum::TOKEN_PACK->value)
            ->sortBy('price');

        $openAiList = OpenAIGenerator::all();

        $currency = currency();

        $lastPrivateDate = false;
        $maxSubscribe = false;

        return view('panel.user.finance.subscriptionPlans', compact(
            'plansSubscriptionMonthly',
            'plansSubscriptionLifetime',
            'plansSubscriptionAnnual',
            'prepaidplans',
            'openAiList',
            'is_active_gateway',
            'activeGateways',
            'activesubid',
            'currency',
            'lastPrivateDate',
            'maxSubscribe',
        ));
    }

    public function creditListPartial(Request $request): JsonResponse
    {
        $cacheKey = $request->get('cache_key', 'credit-list-cache');
        $planId = $request->get('plan_id');
        $userId = $request->get('user_id');
        $user = $userId ? User::find($userId) : auth()->user();

        $plan = $planId ? Plan::getCache(
            static function () use ($planId) {
                return Plan::query()->find($planId);
            },
            '-' . $planId
        ) : null;

        if ($planId) {
            $cacheKey .= '-' . $planId;
            $trackedKeys = Cache::get('credit-list-plan-tracked-keys', []);
            if (! in_array($cacheKey, $trackedKeys, true)) {
                $trackedKeys[] = $cacheKey;
                Cache::forever('credit-list-plan-tracked-keys', $trackedKeys);
            }
        }

        $html = Cache::remember(
            $cacheKey,
            $planId ? null : 2,
            static function () use ($plan, $user) {
                return view('default.components.credit-list-partial', [
                    'categories' => EntityStats::all(),
                    'plan'       => $plan,
                    'user'       => $plan ? null : $user,
                ])->render();
            }
        );

        return response()->json(['html' => $html]);
    }

    public function teamCreditListPartial(Request $request): JsonResponse
    {
        $cacheKey = $request->get('cache_key', 'team-credit-list-cache');
        $team = null;

        if ($teamId = $request->get('team_id')) {
            $team = Team::query()->find($teamId);

            $cacheKey .= '-' . $teamId;
        }

        return response()->json([
            'html' => Cache::remember($cacheKey, 2, static function () use ($team) {
                return view('default.components.team-credit-list-partial', [
                    'categories' => EntityStats::all(),
                    'team'       => $team,
                ])->render();
            }),
        ]);
    }
}
