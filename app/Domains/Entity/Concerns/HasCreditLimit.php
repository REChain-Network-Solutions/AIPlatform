<?php

declare(strict_types=1);

namespace App\Domains\Entity\Concerns;

use App\Domains\Entity\Enums\EntityEnum;
use App\Enums\MagicResponse;
use App\Helpers\Classes\Helper;
use App\Models\Plan;
use App\Models\Setting;
use App\Models\SettingTwo;
use App\Models\Team\Team;
use App\Models\User;
use App\Models\UserUsageCredit;
use Closure;
use Exception;
use Illuminate\Contracts\Auth\Authenticatable;

trait HasCreditLimit
{
    protected ?float $calculatedInputCredit = null;

    public function creditEnum(): EntityEnum
    {
        return $this->enum()->creditBy();
    }

    protected function getPlanWithCredit(): ?Plan
    {
        $this->ensurePlanProvided();

        return $this->plan;
    }

    protected function getTeamWithCredit(): ?Team
    {
        $this->ensureTeamProvided();

        return $this->team;
    }

    protected function getUserWithCredit(): null|User|Authenticatable
    {
        $this->ensureUserProvided();

        return $this->getUser();
    }

    public function getCredit(): array
    {
        if ($this->plan?->exists) {
            return $this->getPlanWithCredit()?->getCredit($this->engine()->slug(), $this->creditKey());
        }
        $user = $this->getUserWithCredit();
        if ($this->team?->exists || $user?->myTeam?->exists) {
            if (! $this->team?->exists) {
                $this->team = $user?->myTeam;
            }

            $memberStat = $user->teamMember;
            if (isset($memberStat) && $memberStat->status !== 'active') {
                // If the user is not an active member of the team, return the user's credit
                return $user?->getCredit($this->engine()->slug(), $this->creditKey()) ?? [];
            }

            $userCredits = $this->getUserWithCredit()?->getCredit($this->engine()->slug(), $this->creditKey());
            $teamCredits = $this->getTeamWithCredit()?->getCredit($this->engine()->slug(), $this->creditKey());

            $mergedCredits = [
                'credit'      => 0,
                'isUnlimited' => false,
            ];

            if (($userCredits['isUnlimited'] ?? false) || ($teamCredits['isUnlimited'] ?? false)) {
                $mergedCredits['isUnlimited'] = true;
                $mergedCredits['credit'] = 0;
            } else {
                $mergedCredits['credit'] = ($userCredits['credit'] ?? 0) + ($teamCredits['credit'] ?? 0);
            }

            return $mergedCredits;
        }

        return $user?->getCredit($this->engine()->slug(), $this->creditKey());
    }

    /**
     * @throws Exception
     */
    public function creditBalance(): float
    {
        return match (config('octane.enabled') || isset($this->team)) {
            true    => $this->getCreditBalance(),
            default => once(function () {
                return $this->getCreditBalance();
            }),
        };
    }

    /**
     * @throws Exception
     */
    public function isUnlimitedCredit(): bool
    {
        return match (config('octane.enabled')) {
            true    => $this->getIsUnlimitedCredit(),
            default => once(function () {
                return $this->getIsUnlimitedCredit();
            }),
        };
    }

    public function getCreditBalance(): float
    {
        $credit = $this->getCredit()['credit'];

        if (is_string($credit)) {
            $credit = (float) $credit;
        }

        $aiFinances = app('ai_chat_model_plan');

        $engineDefaultModels = $this->engine()->getDefaultModels(Setting::getCache(), SettingTwo::getCache());
        $model = $this->model(config('octane.enabled'));
        if (
            $model && ! $model->is_selected &&
            ! in_array($model->key, $engineDefaultModels, true) &&
            ! in_array($model->id, $aiFinances, true)
        ) {
            return 0;
        }

        return $credit;
    }

    /**
     * @throws Exception
     */
    public function getIsUnlimitedCredit(): bool
    {
        $aiFinances = app('ai_chat_model_plan');

        $engineDefaultModels = $this->engine()->getDefaultModels(Setting::getCache(), SettingTwo::getCache());

        $model = $this->model(config('octane.enabled'));

        if (
            $model && ! $model->is_selected &&
            ! in_array($model->key, $engineDefaultModels, true) &&
            ! in_array($model->id, $aiFinances, true)
        ) {
            return false;
        }

        return $this->getCredit()['isUnlimited'];
    }

    /**
     * @throws Exception
     */
    public function hasCreditBalance(): bool
    {
        if ($this->guest) {
            return $this->guestHasAttempts();
        }

        return $this->creditBalance() > 0 || $this->isUnlimitedCredit();
    }

    public function guestHasAttempts(): bool
    {
        $clientIp = Helper::getRequestIp();
        $rateLimiter = new \App\Helpers\Classes\RateLimiter\RateLimiter('guest-chat-attempt', (int) setting('guest_user_daily_message_limit', '10'));
        if ($rateLimiter->attempt($clientIp)) {
            return true;
        }

        return false;
    }

    /**
     * @throws Exception
     */
    public function redirectIfNoCreditBalance(): void
    {
        if ($this->hasCreditBalance()) {
            return;
        }

        MagicResponse::NO_CREDITS_LEFT->exception();
    }

    /**
     * @throws Exception
     */
    public function setCredit(float $value = 1.00): bool
    {
        return $this->updateUserCredit($value, function ($creditBalance, $credit) {
            return $credit;
        }, skipCalculatedCredit: true);
    }

    /**
     * @throws Exception
     */
    public function setDefaultCreditForDemo(): bool
    {
        if ($this->getUserWithCredit()?->isAdmin()) {
            $this->setAsUnlimited();
        }

        return $this->setCredit($this->creditEnum()->defaultCreditForDemo());
    }

    public function setAsUnlimited(bool $unlimited = true): bool
    {
        $user = $this->getUserWithCredit();
        $creditKey = $this->creditKey();
        $engineKey = $this->engine()->slug();

        $creditsArr = $user?->entity_credits;

        $creditsArr[$engineKey][$creditKey] = [
            'credit'              => $creditsArr[$engineKey][$creditKey]['credit'] ?? 0.0,
            'isUnlimited'         => $unlimited,
        ];

        return $user?->update([
            'entity_credits' => $creditsArr,
        ]);
    }

    /**
     * @throws Exception
     */
    public function increaseCredit(float $value = 1.00): bool
    {
        return $this->updateUserCredit($value, function ($creditBalance, $credit) {
            return $creditBalance + $credit;
        });
    }

    /**
     * @throws Exception
     */
    public function decreaseCredit(float $value = 1.00): bool
    {
        if ($this->guest || $this->isUnlimitedCredit()) {
            return true;
        }

        $unitPrice = EntityEnum::fromSlug($this->enum()->slug())->unitPrice();
        $currentSpend = $value * $unitPrice;
        setting(['total_spend' => number_format((setting('total_spend', 0) + $currentSpend), 2)])->save();

        UserUsageCredit::create([
            'user_id'     => $this->getUser()->id,
            'model_key'   => $this->enum()->slug(),
            'credit'      => $value,
            'unit_price'  => $unitPrice,
            'total'       => $value * $unitPrice,
        ]);

        return $this->updateUserCredit($value, function ($creditBalance, $credit) {
            return max(0, $creditBalance - $credit);
        });
    }

    /**
     * @throws Exception
     */
    private function updateUserCredit(float $value, Closure $callback, bool $skipCalculatedCredit = false): bool
    {
        $user = $this->getUserWithCredit();
        $team = $this->team;

        if ($skipCalculatedCredit) {
            $credit = $value;
        } else {
            $credit = $this->calculatedInputCredit ?: $value;
        }

        $creditKey = $this->creditKey();

        $engineKey = $this->engine()->slug();

        $isTeamCredit = isset($team) && $team->exists;

        $target = $isTeamCredit ? $team : $user;
        $creditsArr = $target && isset($target->entity_credits) ? $target->entity_credits : User::getFreshCredits();

        $creditsArr[$engineKey][$creditKey] = [
            'credit'      => $callback($this->creditBalance(), $credit),
            'isUnlimited' => $creditsArr[$engineKey][$creditKey]['isUnlimited'] ?? false,
        ];

        return $target?->update([
            'entity_credits' => $creditsArr,
        ]);
    }

    public function getCreditIndex(): float
    {
        return $this->creditEnum()->creditIndex();
    }

    public function getCalculatedInputCredit(): float
    {
        return $this->calculatedInputCredit;
    }

    /**
     * @throws Exception
     */
    public function hasCreditBalanceForInput(): bool
    {
        if ($this->isUnlimitedCredit()) {
            return true;
        }

        return $this->creditBalance() > $this->getCalculatedInputCredit();
    }

    public function setCalculatedInputCredit($value = 0.0): static
    {
        $this->calculatedInputCredit = $value;

        return $this;
    }
}
