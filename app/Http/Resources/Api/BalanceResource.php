<?php

declare(strict_types=1);

namespace App\Http\Resources\Api;

use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;

class BalanceResource extends JsonResource
{
    public function toArray(Request $request): array
    {
        $plan = $this->activePlan();

        return [
            'balance'            => (float) $this->shared_credits,
            'credit_system_type' => $this->credit_system_type,
            'auto_topup'         => (bool) $this->shared_credit_auto_topup,
            'plan'               => $plan ? [
                'name'            => $plan->name,
                'credits_amount'  => (float) $plan->shared_credits_amount,
            ] : null,
        ];
    }
}
