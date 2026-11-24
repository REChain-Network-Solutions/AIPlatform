<?php

declare(strict_types=1);

namespace App\Http\Resources\Api;

use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;

class TransactionResource extends JsonResource
{
    public function toArray(Request $request): array
    {
        return [
            'id'            => $this->id,
            'entity_key'    => $this->entity_key,
            'action_type'   => $this->action_type,
            'amount'        => (float) $this->amount,
            'balance_after' => (float) $this->balance_after,
            'unit_cost'     => (float) $this->unit_cost,
            'quantity'      => (float) $this->quantity,
            'description'   => $this->description,
            'created_at'    => $this->created_at?->toIso8601String(),
        ];
    }
}
