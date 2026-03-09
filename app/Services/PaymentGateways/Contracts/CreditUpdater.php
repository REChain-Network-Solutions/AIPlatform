<?php

namespace App\Services\PaymentGateways\Contracts;

use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Http\Controllers\Team\TeamController;
use App\Models\Plan;
use App\Models\User;
use Exception;

trait CreditUpdater
{
    public static function creditIncreaseSubscribePlan(?User $user, ?Plan $plan): void
    {
        $team = null;
        $isTeamPlan = $plan?->getAttribute('is_team_plan') ?? false;
        if ($isTeamPlan) {
            $team = app(TeamController::class)->getTeam($user);
        }

        $modelsCredit = $plan?->getAttribute('ai_models');
        foreach ($modelsCredit as $modelsGroup) {
            foreach ($modelsGroup as $model => $credit) {
                $driver = $isTeamPlan ? Entity::driver(EntityEnum::fromSlug($model))->forTeam($team) : Entity::driver(EntityEnum::fromSlug($model))->forUser($user);
                if ($plan?->getAttribute('reset_credits_on_renewal')) {
                    $driver->setCredit($credit['credit']);
                } else {
                    $driver->increaseCredit($credit['credit']);
                }
                $driver->setAsUnlimited($credit['isUnlimited']);
            }
        }
    }

    /**
     * @throws Exception
     */
    public static function creditDecreaseCancelPlan(User $user, Plan $plan): void
    {
        $team = null;
        $isTeamPlan = $plan->getAttribute('is_team_plan') ?? false;
        if ($isTeamPlan) {
            $team = app(TeamController::class)->getTeam($user);
        }

        if ((bool) setting('soft_plan_cancellation', false)) {
            return;
        }
        $modelsCredit = $plan->getAttribute('ai_models');
        foreach ($modelsCredit as $modelsGroup) {
            foreach ($modelsGroup as $model => $credit) {
                $driver = $isTeamPlan ? Entity::driver(EntityEnum::fromSlug($model))->forTeam($team) : Entity::driver(EntityEnum::fromSlug($model))->forUser($user);
                $driver->setAsUnlimited(false);
                $driver->decreaseCredit($credit['credit']);
            }
        }
    }
}
