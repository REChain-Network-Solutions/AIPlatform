<?php

declare(strict_types=1);

namespace App\Http\Resources\Api;

use App\Domains\Entity\Enums\EntityEnum;
use App\Services\SharedCredit\SharedCreditService;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;
use Throwable;

class CostResource extends JsonResource
{
    public function toArray(Request $request): array
    {
        /** @var EntityEnum $entity */
        $entity = $this->resource;
        $service = app(SharedCreditService::class);
        $override = $service->getCostOverride($entity->slug());

        $featureType = 'unknown';

        try {
            $featureType = $entity->tokenType()->value;
        } catch (Throwable) {
        }

        return [
            'entity_key'          => $entity->slug(),
            'label'               => $entity->label(),
            'feature_type'        => $featureType,
            'shared_credit_index' => $entity->sharedCreditIndex(),
            'effective_cost'      => $service->resolveUnitCost($entity),
            'has_db_override'     => $override !== null,
        ];
    }
}
