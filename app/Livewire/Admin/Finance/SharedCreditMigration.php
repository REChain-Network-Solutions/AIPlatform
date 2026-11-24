<?php

declare(strict_types=1);

namespace App\Livewire\Admin\Finance;

use App\Domains\Entity\Enums\EntityEnum;
use App\Livewire\Concerns\WithSteps;
use App\Models\Plan;
use App\Models\SharedCreditTransaction;
use App\Models\Subscriptions;
use App\Models\User;
use Illuminate\Contracts\View\View;
use Illuminate\Support\Carbon;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;
use Livewire\Component;
use Throwable;

class SharedCreditMigration extends Component
{
    use WithSteps;

    protected int $maxStep = 5;

    /** @var array<int, array{id: int, name: string, frequency: string, price: float, user_count: int, credit_system_type: string}> */
    public array $plans = [];

    public array $selectedPlanIds = [];

    /** @var array<int, array{shared_credits_amount: float, unlimited_models: array}> */
    public array $planConfigs = [];

    public string $migrationStrategy = 'immediate';

    public bool $executed = false;

    public array $results = [];

    public string $migrationBatchId = '';

    public function mount(): void
    {
        $this->step = 1;
        $this->loadPlans();
    }

    public function stepTitle(int $num): string
    {
        return match ($num) {
            1 => 'Analysis',
            2 => 'Configure',
            3 => 'Users',
            4 => 'Execute',
            5 => 'Notify',
        };
    }

    public function getPercent(): int
    {
        return match ($this->currentStep()) {
            1       => 20,
            2       => 40,
            3       => 60,
            4       => 80,
            default => 100,
        };
    }

    public function nextStep(): void
    {
        if ($this->currentStepIs(1)) {
            if (empty($this->selectedPlanIds)) {
                $this->addError('selectedPlanIds', __('Please select at least one plan to migrate.'));

                return;
            }
            $this->buildPlanConfigs();
        }

        if ($this->currentStepIs(4) && ! $this->executed) {
            return;
        }

        if (! $this->hasNextStep()) {
            return;
        }

        $this->toNextStep();
    }

    public function togglePlan(int $planId): void
    {
        if (in_array($planId, $this->selectedPlanIds)) {
            $this->selectedPlanIds = array_values(array_diff($this->selectedPlanIds, [$planId]));
        } else {
            $this->selectedPlanIds[] = $planId;
        }
    }

    public function selectAllPlans(): void
    {
        $this->selectedPlanIds = collect($this->plans)->pluck('id')->all();
    }

    public function deselectAllPlans(): void
    {
        $this->selectedPlanIds = [];
    }

    public function executeMigration(): void
    {
        if ($this->executed) {
            return;
        }

        $this->migrationBatchId = 'migration_' . now()->format('Ymd_His') . '_' . uniqid();

        try {
            DB::beginTransaction();

            $results = [
                'plans_migrated'  => 0,
                'users_migrated'  => 0,
                'errors'          => [],
                'plan_details'    => [],
            ];

            foreach ($this->selectedPlanIds as $planId) {
                $plan = Plan::query()->find($planId);
                if (! $plan) {
                    $results['errors'][] = "Plan #{$planId} not found.";

                    continue;
                }

                $config = $this->planConfigs[$planId] ?? null;
                if (! $config) {
                    $results['errors'][] = "No configuration for plan '{$plan->name}'.";

                    continue;
                }

                $oldSystemType = $plan->credit_system_type;
                $plan->credit_system_type = 'shared';
                $plan->shared_credits_amount = $config['shared_credits_amount'];
                $plan->save();

                $results['plans_migrated']++;
                $results['plan_details'][] = [
                    'id'                 => $plan->id,
                    'name'               => $plan->name,
                    'old_type'           => $oldSystemType,
                    'shared_credits'     => $config['shared_credits_amount'],
                    'users_affected'     => 0,
                ];

                if ($this->migrationStrategy === 'immediate') {
                    $usersAffected = $this->migrateUsersForPlan($plan, $config, $results);
                    $results['plan_details'][array_key_last($results['plan_details'])]['users_affected'] = $usersAffected;
                }
            }

            DB::commit();

            Plan::forgetCache();
            $this->results = $results;
            $this->executed = true;
        } catch (Throwable $e) {
            DB::rollBack();
            Log::error('Credit migration failed', [
                'error' => $e->getMessage(),
                'trace' => $e->getTraceAsString(),
            ]);
            $this->addError('migration', __('Migration failed: ') . $e->getMessage());
        }
    }

    public function rollbackMigration(): void
    {
        if (! $this->executed || empty($this->migrationBatchId)) {
            return;
        }

        try {
            DB::beginTransaction();

            $transactions = SharedCreditTransaction::query()
                ->where('action_type', 'migration')
                ->where('metadata->batch_id', $this->migrationBatchId)
                ->get();

            foreach ($transactions as $transaction) {
                $user = User::query()->find($transaction->user_id);
                if (! $user) {
                    continue;
                }

                $metadata = $transaction->metadata ?? [];
                $user->credit_system_type = $metadata['old_credit_system_type'] ?? 'separated';
                $user->shared_credits = max(0, ($user->shared_credits ?? 0) - $transaction->amount);
                $user->save();

                $transaction->delete();
            }

            foreach ($this->selectedPlanIds as $planId) {
                $plan = Plan::query()->find($planId);
                if (! $plan) {
                    continue;
                }

                $config = $this->planConfigs[$planId] ?? null;
                $plan->credit_system_type = $config['old_credit_system_type'] ?? 'separated';
                $plan->shared_credits_amount = 0;
                $plan->save();
            }

            DB::commit();

            Plan::forgetCache();
            $this->executed = false;
            $this->results = [];
            $this->migrationBatchId = '';
            $this->step = 1;
            $this->loadPlans();
        } catch (Throwable $e) {
            DB::rollBack();
            Log::error('Credit migration rollback failed', [
                'error' => $e->getMessage(),
            ]);
            $this->addError('rollback', __('Rollback failed: ') . $e->getMessage());
        }
    }

    public function getSelectedPlansProperty(): Collection
    {
        return collect($this->plans)->filter(
            fn (array $plan) => in_array($plan['id'], $this->selectedPlanIds)
        )->values();
    }

    public function getActiveUserCountProperty(): int
    {
        if (empty($this->selectedPlanIds)) {
            return 0;
        }

        return Subscriptions::query()
            ->whereIn('plan_id', $this->selectedPlanIds)
            ->whereIn('stripe_status', $this->activeStatuses())
            ->distinct('user_id')
            ->count('user_id');
    }

    public function render(): View
    {
        return view('livewire.admin.finance.shared-credit-migration');
    }

    private function loadPlans(): void
    {
        $this->plans = Plan::query()
            ->where('credit_system_type', '!=', 'shared')
            ->orWhereNull('credit_system_type')
            ->orderBy('name')
            ->get()
            ->map(function (Plan $plan) {
                $userCount = Subscriptions::query()
                    ->where('plan_id', $plan->id)
                    ->whereIn('stripe_status', $this->activeStatuses())
                    ->distinct('user_id')
                    ->count('user_id');

                return [
                    'id'                 => $plan->id,
                    'name'               => $plan->name,
                    'frequency'          => $plan->frequency ?? 'monthly',
                    'price'              => (float) $plan->price,
                    'user_count'         => $userCount,
                    'credit_system_type' => $plan->credit_system_type ?? 'separated',
                ];
            })
            ->all();
    }

    private function buildPlanConfigs(): void
    {
        foreach ($this->selectedPlanIds as $planId) {
            if (isset($this->planConfigs[$planId])) {
                continue;
            }

            $plan = Plan::query()->find($planId);
            if (! $plan) {
                continue;
            }

            $suggestedAmount = $this->calculateSuggestedAmount($plan);
            $unlimitedModels = $this->getUnlimitedModels($plan);

            $this->planConfigs[$planId] = [
                'shared_credits_amount'  => round($suggestedAmount, 2),
                'unlimited_models'       => $unlimitedModels,
                'old_credit_system_type' => $plan->credit_system_type ?? 'separated',
                'breakdown'              => $this->getCreditBreakdown($plan),
            ];
        }
    }

    private function calculateSuggestedAmount(Plan $plan): float
    {
        $total = 0.0;
        $aiModels = $plan->ai_models ?? [];

        foreach ($aiModels as $engineModels) {
            if (! is_array($engineModels)) {
                continue;
            }

            foreach ($engineModels as $slug => $config) {
                if (! is_array($config)) {
                    continue;
                }

                if ($config['isUnlimited'] ?? false) {
                    continue;
                }

                $credit = (float) ($config['credit'] ?? 0);
                if ($credit <= 0) {
                    continue;
                }

                try {
                    $entity = EntityEnum::fromSlug($slug);
                    $total += $credit * $entity->sharedCreditIndex();
                } catch (Throwable) {
                    continue;
                }
            }
        }

        return $total;
    }

    private function getUnlimitedModels(Plan $plan): array
    {
        $unlimited = [];
        $aiModels = $plan->ai_models ?? [];

        foreach ($aiModels as $engineModels) {
            if (! is_array($engineModels)) {
                continue;
            }

            foreach ($engineModels as $slug => $config) {
                if (! is_array($config)) {
                    continue;
                }

                if ($config['isUnlimited'] ?? false) {
                    try {
                        $entity = EntityEnum::fromSlug($slug);
                        $unlimited[] = [
                            'slug'  => $slug,
                            'label' => $entity->label(),
                        ];
                    } catch (Throwable) {
                        $unlimited[] = [
                            'slug'  => $slug,
                            'label' => $slug,
                        ];
                    }
                }
            }
        }

        return $unlimited;
    }

    private function getCreditBreakdown(Plan $plan): array
    {
        $breakdown = [];
        $aiModels = $plan->ai_models ?? [];

        foreach ($aiModels as $engineKey => $engineModels) {
            if (! is_array($engineModels)) {
                continue;
            }

            foreach ($engineModels as $slug => $config) {
                if (! is_array($config)) {
                    continue;
                }

                $credit = (float) ($config['credit'] ?? 0);
                $isUnlimited = $config['isUnlimited'] ?? false;

                if ($credit <= 0 && ! $isUnlimited) {
                    continue;
                }

                try {
                    $entity = EntityEnum::fromSlug($slug);
                    $index = $entity->sharedCreditIndex();
                    $breakdown[] = [
                        'label'        => $entity->label(),
                        'credit'       => $credit,
                        'index'        => $index,
                        'shared_value' => $isUnlimited ? null : round($credit * $index, 2),
                        'is_unlimited' => $isUnlimited,
                    ];
                } catch (Throwable) {
                    continue;
                }
            }
        }

        return $breakdown;
    }

    private function migrateUsersForPlan(Plan $plan, array $config, array &$results): int
    {
        $usersAffected = 0;

        $subscriptions = Subscriptions::query()
            ->where('plan_id', $plan->id)
            ->whereIn('stripe_status', $this->activeStatuses())
            ->get();

        foreach ($subscriptions as $subscription) {
            $user = User::query()->find($subscription->user_id);
            if (! $user) {
                continue;
            }

            $sharedCredits = $config['shared_credits_amount'];

            if ($subscription->ends_at) {
                $endsAt = Carbon::parse($subscription->ends_at);
                $createdAt = Carbon::parse($subscription->created_at);
                $totalDays = max(1, $createdAt->diffInDays($endsAt));
                $daysLeft = max(0, now()->diffInDays($endsAt, false));
                $ratio = min(1, $daysLeft / $totalDays);
                $sharedCredits = round($sharedCredits * $ratio, 2);
            }

            $oldCreditSystemType = $user->credit_system_type;
            $oldSharedCredits = (float) ($user->shared_credits ?? 0);

            $user->credit_system_type = 'shared';
            $user->shared_credits = $oldSharedCredits + $sharedCredits;
            $user->save();

            SharedCreditTransaction::query()->create([
                'user_id'       => $user->id,
                'action_type'   => 'migration',
                'amount'        => $sharedCredits,
                'balance_after' => $user->shared_credits,
                'description'   => "Migrated from plan '{$plan->name}' (separated → shared)",
                'metadata'      => [
                    'batch_id'               => $this->migrationBatchId,
                    'plan_id'                => $plan->id,
                    'plan_name'              => $plan->name,
                    'old_credit_system_type' => $oldCreditSystemType,
                    'old_shared_credits'     => $oldSharedCredits,
                    'pro_rata_ratio'         => $subscription->ends_at ? ($ratio ?? 1) : 1,
                ],
            ]);

            $usersAffected++;
            $results['users_migrated']++;
        }

        return $usersAffected;
    }

    private function activeStatuses(): array
    {
        return [
            'active',
            'trialing',
            'bank_approved',
            'banktransfer_approved',
            'bank_renewed',
            'free_approved',
            'stripe_approved',
            'paypal_approved',
            'iyzico_approved',
            'paystack_approved',
        ];
    }
}
