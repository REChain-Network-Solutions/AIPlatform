<?php

declare(strict_types=1);

namespace App\Http\Controllers;

use App\Services\GeneratorMappingService;
use App\Services\SharedCredit\SharedCreditService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;

class SharedCreditPreviewController extends Controller
{
    public function __construct(
        private SharedCreditService $sharedCreditService,
        private GeneratorMappingService $generatorMappingService,
    ) {}

    public function balance(Request $request): JsonResponse
    {
        $user = $request->user();
        $team = $user->myTeam;
        $teamBalance = ($team && $team->exists && $team->isSharedCreditTeam())
            ? (float) $team->shared_credits
            : null;

        return response()->json([
            'balance'      => (float) $user->shared_credits,
            'team_balance' => $teamBalance,
        ]);
    }

    public function preview(Request $request): JsonResponse
    {
        $request->validate([
            'generator'  => 'required|string',
            'quantity'   => 'numeric|min:0|max:100000',
            'action'     => 'nullable|string',
        ]);

        $user = $request->user();

        if (! $this->sharedCreditService->isEnabled() || ! $user->isSharedCreditUser()) {
            return response()->json([
                'success'               => true,
                'is_shared_credit_user' => false,
            ]);
        }

        $entity = $this->generatorMappingService->mapGeneratorToEntity(
            $request->input('generator'),
            $request->input('action')
        );

        if (! $entity) {
            return response()->json([
                'success' => false,
                'message' => __('Unknown generator.'),
            ], 400);
        }

        $unitCost = $this->sharedCreditService->resolveUnitCost($entity);
        $quantity = (float) $request->input('quantity', 1);
        $totalCost = $unitCost * $quantity;
        $balance = $this->sharedCreditService->getBalance($user);
        $allowNegative = (bool) setting('shared_credit_allow_negative', false);

        return response()->json([
            'success'               => true,
            'is_shared_credit_user' => true,
            'data'                  => [
                'entity'         => $entity->value,
                'unit_cost'      => $unitCost,
                'quantity'       => $quantity,
                'total_cost'     => $totalCost,
                'balance'        => $balance,
                'has_balance'    => $allowNegative || $balance >= $totalCost,
                'allow_negative' => $allowNegative,
            ],
        ]);
    }
}
