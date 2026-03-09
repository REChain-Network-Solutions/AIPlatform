<?php

namespace App\Services\Finance;

use App\Enums\Plan\FrequencyEnum;
use App\Enums\Plan\TypeEnum;
use App\Models\Plan;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Cache;

class PlanService
{
    public const ACTIVE_PLANS_CACHE_KEY = 'active_plans_cache_v1';

    public const CREDIT_LIST_PLAN_TRACKED_KEYS = 'credit-list-plan-tracked-keys';

    private ?Collection $plans = null;

    private function getPlans(): Collection
    {
        if ($this->plans === null) {
            $this->plans = Cache::rememberForever(self::ACTIVE_PLANS_CACHE_KEY, static function () {
                return Plan::where('active', true)
                    ->where('hidden', false)
                    ->orderBy('price')
                    ->get();
            });
        }

        return $this->plans;
    }

    public function getSubscriptionPlans(): Collection
    {
        return $this->getPlans()->where('type', TypeEnum::SUBSCRIPTION->value);
    }

    public function getMonthlySubscriptions(): Collection
    {
        return $this->getSubscriptionPlans()
            ->where('frequency', FrequencyEnum::MONTHLY->value);
    }

    public function getLifetimeSubscriptions(): Collection
    {
        return $this->getSubscriptionPlans()
            ->filter(fn ($plan) => in_array($plan->frequency, [
                FrequencyEnum::LIFETIME_YEARLY->value,
                FrequencyEnum::LIFETIME_MONTHLY->value,
            ], true));
    }

    public function getAnnualSubscriptions(): Collection
    {
        return $this->getSubscriptionPlans()
            ->where('frequency', FrequencyEnum::YEARLY->value);
    }

    public function getPrepaidPlans(): Collection
    {
        return $this->getPlans()->where('type', TypeEnum::TOKEN_PACK->value);
    }

    public static function clearCache(): void
    {
        Cache::forget(self::ACTIVE_PLANS_CACHE_KEY);
        Cache::forget(self::CREDIT_LIST_PLAN_TRACKED_KEYS);
        Plan::forgetCache();
    }
}
