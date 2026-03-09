<?php

namespace App\Helpers\Classes;

use App\Models\Plan;
use App\Services\Common\MenuService;
use Illuminate\Support\Facades\Auth;

class PlanHelper
{
    public static function userPlanAiModel(): ?string
    {
        return self::userPlan()?->default_ai_model;
    }

    public static function userPlan()
    {
        if (Auth::check()) {
            $userId = Auth::id();
            $subscription = getCurrentActiveSubscription($userId);
            if ($subscription) {
                return Plan::getCache(
                    static function () use ($subscription) {
                        return Plan::query()->where('id', $subscription->plan_id)->first();
                    },
                    '_user_plan_' . $userId . '_' . $subscription->plan_id
                );
            }
        }

        return null;
    }

    public static function planMenuCheck($plan, ?string $key = null): bool
    {
        if (! $plan) {
            return true;
        }

        if (! $key) {
            return true;
        }

        $dataAiTools = array_map(static function ($item) {
            return $item['key'];
        }, MenuService::planAiToolsMenu());

        $dataFeature = array_map(static function ($item) {
            return $item['key'];
        }, MenuService::planFeatureMenu());

        $data = array_merge($dataAiTools, $dataFeature);

        if (! in_array($key, $data, true)) {
            return true;
        }

        $plan_ai_tools = (array) $plan->plan_ai_tools;

        $plan_features = (array) $plan->plan_features;

        $checkArray = array_merge($plan_ai_tools, $plan_features);

        if ($checkArray) {
            return array_key_exists($key, $checkArray) && $checkArray[$key];
        }

        return false;
    }
}
