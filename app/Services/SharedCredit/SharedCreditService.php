<?php

declare(strict_types=1);

namespace App\Services\SharedCredit;

use App\Domains\Entity\Enums\EntityEnum;
use App\Models\SharedCreditCost;
use App\Models\SharedCreditTransaction;
use App\Models\Team\Team;
use App\Models\User;
use App\Notifications\LowBalanceNotification;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;
use Throwable;

class SharedCreditService
{
    public function isEnabled(): bool
    {
        return (bool) setting('shared_credit_system_enabled', false);
    }

    public function getBalance(User $user): float
    {
        return (float) $user->shared_credits;
    }

    public function hasBalance(User $user, float $amount): bool
    {
        return $this->getBalance($user) >= $amount;
    }

    public function resolveUnitCost(EntityEnum $entity): float
    {
        $override = $this->getCostOverride($entity->slug());

        return $override ? $override->base_cost : $entity->sharedCreditIndex();
    }

    public function deduct(User $user, EntityEnum $entity, float $quantity, ?array $metadata = null): ?SharedCreditTransaction
    {
        $unitCost = $this->resolveUnitCost($entity);
        $amount = $quantity * $unitCost;

        $transaction = DB::transaction(function () use ($user, $entity, $quantity, $unitCost, $amount, $metadata) {
            $freshUser = User::query()->lockForUpdate()->find($user->id);

            if ($freshUser->shared_credits < $amount) {
                return null;
            }

            $freshUser->shared_credits = max(0, $freshUser->shared_credits - $amount);
            $freshUser->save();

            $user->shared_credits = $freshUser->shared_credits;

            return $this->logTransaction(
                user: $freshUser,
                actionType: 'deduct',
                amount: -$amount,
                balanceAfter: $freshUser->shared_credits,
                entity: $entity,
                unitCost: $unitCost,
                quantity: $quantity,
                metadata: $metadata,
            );
        });

        if ($transaction) {
            $this->checkPostDeductionHooks($user->refresh());
        }

        return $transaction;
    }

    public function deductWithTeam(User $user, Team $team, EntityEnum $entity, float $quantity, ?array $metadata = null): ?SharedCreditTransaction
    {
        $unitCost = $this->resolveUnitCost($entity);
        $amount = $quantity * $unitCost;

        $transaction = DB::transaction(function () use ($user, $team, $entity, $quantity, $unitCost, $amount, $metadata) {
            $freshUser = User::query()->lockForUpdate()->find($user->id);
            $freshTeam = Team::query()->lockForUpdate()->find($team->id);

            // Daily limit check
            $memberStat = $freshUser->teamMember;
            if ($memberStat && $memberStat->daily_shared_credit_limit !== null) {
                $todaySpent = SharedCreditTransaction::query()
                    ->where('user_id', $freshUser->id)
                    ->where('action_type', 'deduct')
                    ->whereDate('created_at', now()->toDateString())
                    ->sum(DB::raw('ABS(amount)'));

                if (($todaySpent + $amount) > $memberStat->daily_shared_credit_limit) {
                    return null;
                }
            }

            $totalAvailable = $freshUser->shared_credits + $freshTeam->shared_credits;

            if ($totalAvailable < $amount) {
                return null;
            }

            // Deduct from user first, remainder from team
            $userDeduction = min($freshUser->shared_credits, $amount);
            $teamDeduction = $amount - $userDeduction;

            $freshUser->shared_credits = max(0, $freshUser->shared_credits - $userDeduction);
            $freshUser->save();

            if ($teamDeduction > 0) {
                $freshTeam->shared_credits = max(0, $freshTeam->shared_credits - $teamDeduction);
                $freshTeam->save();
            }

            $user->shared_credits = $freshUser->shared_credits;

            $source = $teamDeduction > 0 ? 'team_split' : 'user';
            $mergedMeta = array_merge($metadata ?? [], [
                'source'         => $source,
                'user_deduction' => $userDeduction,
                'team_deduction' => $teamDeduction,
                'team_id'        => $freshTeam->id,
            ]);

            return $this->logTransaction(
                user: $freshUser,
                actionType: 'deduct',
                amount: -$amount,
                balanceAfter: $freshUser->shared_credits,
                entity: $entity,
                unitCost: $unitCost,
                quantity: $quantity,
                metadata: $mergedMeta,
            );
        });

        if ($transaction) {
            $this->checkPostDeductionHooks($user->refresh());
        }

        return $transaction;
    }

    public function topUp(User $user, float $amount, string $description, ?EntityEnum $entity = null, string $actionType = 'top_up'): SharedCreditTransaction
    {
        return DB::transaction(function () use ($user, $amount, $description, $entity, $actionType) {
            $freshUser = User::query()->lockForUpdate()->find($user->id);
            $freshUser->shared_credits += $amount;
            $freshUser->save();

            $user->shared_credits = $freshUser->shared_credits;

            return $this->logTransaction(
                user: $freshUser,
                actionType: $actionType,
                amount: $amount,
                balanceAfter: $freshUser->shared_credits,
                entity: $entity,
                description: $description,
            );
        });
    }

    public function logTransaction(
        User $user,
        string $actionType,
        float $amount,
        float $balanceAfter,
        ?EntityEnum $entity = null,
        float $unitCost = 0,
        float $quantity = 0,
        ?string $quality = null,
        ?array $metadata = null,
        ?string $description = null,
    ): SharedCreditTransaction {
        return SharedCreditTransaction::create([
            'user_id'       => $user->id,
            'entity_key'    => $entity?->slug(),
            'engine_key'    => $entity?->engine()->value,
            'feature_type'  => $entity ? $this->resolveFeatureType($entity) : null,
            'action_type'   => $actionType,
            'amount'        => $amount,
            'balance_after' => $balanceAfter,
            'unit_cost'     => $unitCost,
            'quantity'      => $quantity,
            'quality'       => $quality,
            'metadata'      => $metadata,
            'description'   => $description,
        ]);
    }

    public function getHistory(User $user, ?string $actionType = null, int $limit = 50): Collection
    {
        $query = $user->sharedCreditTransactions()->latest();

        if ($actionType) {
            $query->where('action_type', $actionType);
        }

        return $query->limit($limit)->get();
    }

    public function getCostOverride(string $entityKey): ?SharedCreditCost
    {
        return Cache::remember(
            "shared_credit_cost:{$entityKey}",
            300,
            fn () => SharedCreditCost::query()->active()->where('entity_key', $entityKey)->first()
        );
    }

    private function checkPostDeductionHooks(User $user): void
    {
        // Low balance notification
        $planCredits = $user->activePlan()?->shared_credits_amount ?? 0;
        $thresholdPercent = (int) setting('shared_credit_low_balance_threshold', 20);
        if ($planCredits > 0) {
            $percent = ($user->shared_credits / $planCredits) * 100;
            if ($percent <= $thresholdPercent) {
                $user->notify(new LowBalanceNotification($user->shared_credits, (float) $planCredits));
            }
        }
    }

    private function resolveFeatureType(EntityEnum $entity): string
    {
        try {
            return $entity->tokenType()->value;
        } catch (Throwable) {
            return 'unknown';
        }
    }
}
