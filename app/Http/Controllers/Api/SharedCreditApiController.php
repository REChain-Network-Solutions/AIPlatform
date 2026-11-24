<?php

declare(strict_types=1);

namespace App\Http\Controllers\Api;

use App\Domains\Entity\Enums\EntityEnum;
use App\Http\Controllers\Controller;
use App\Http\Resources\Api\BalanceResource;
use App\Http\Resources\Api\CostResource;
use App\Http\Resources\Api\TransactionResource;
use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\AnonymousResourceCollection;

class SharedCreditApiController extends Controller
{
    public function balance(Request $request): BalanceResource
    {
        return new BalanceResource($request->user());
    }

    public function history(Request $request): AnonymousResourceCollection
    {
        $transactions = $request->user()
            ->sharedCreditTransactions()
            ->latest()
            ->paginate($request->integer('per_page', 15));

        return TransactionResource::collection($transactions);
    }

    public function cost(string $entity): CostResource
    {
        $entityEnum = EntityEnum::fromSlug($entity);

        return new CostResource($entityEnum);
    }
}
