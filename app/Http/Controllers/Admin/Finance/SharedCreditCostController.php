<?php

declare(strict_types=1);

namespace App\Http\Controllers\Admin\Finance;

use App\Domains\Entity\Enums\EntityEnum;
use App\Http\Controllers\Controller;
use App\Models\SharedCreditCost;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Illuminate\View\View;
use Throwable;

class SharedCreditCostController extends Controller
{
    public function index(): View
    {
        $entities = EntityEnum::listableCases();
        $overrides = SharedCreditCost::all()->keyBy('entity_key');

        $entityData = $entities->map(fn (EntityEnum $e) => [
            'slug'        => $e->slug(),
            'label'       => $e->label(),
            'engine'      => $e->engine()->value,
            'hasOverride' => isset($overrides[$e->slug()]),
        ])->values();

        return view('panel.admin.finance.shared-credit-costs.index', compact('entities', 'overrides', 'entityData'));
    }

    public function store(Request $request): RedirectResponse
    {
        $validated = $request->validate([
            'entity_key' => 'required|string|unique:shared_credit_costs,entity_key',
            'base_cost'  => 'required|numeric|min:0',
            'is_active'  => 'nullable|boolean',
        ]);

        $entity = EntityEnum::fromSlug($validated['entity_key']);

        SharedCreditCost::create([
            'entity_key'   => $validated['entity_key'],
            'engine_key'   => $entity->engine()->value,
            'feature_type' => $this->resolveFeatureType($entity),
            'base_cost'    => $validated['base_cost'],
            'is_active'    => $validated['is_active'] ?? true,
        ]);

        Cache::forget("shared_credit_cost:{$validated['entity_key']}");

        return back()->with(['message' => __('Cost override created successfully.'), 'type' => 'success']);
    }

    public function update(Request $request, SharedCreditCost $cost): RedirectResponse
    {
        $validated = $request->validate([
            'base_cost' => 'required|numeric|min:0',
            'is_active' => 'nullable|boolean',
        ]);

        $cost->update([
            'base_cost' => $validated['base_cost'],
            'is_active' => $request->has('is_active'),
        ]);

        Cache::forget("shared_credit_cost:{$cost->entity_key}");

        return back()->with(['message' => __('Cost override updated successfully.'), 'type' => 'success']);
    }

    public function destroy(SharedCreditCost $cost): RedirectResponse
    {
        $entityKey = $cost->entity_key;
        $cost->delete();

        Cache::forget("shared_credit_cost:{$entityKey}");

        return back()->with(['message' => __('Cost override removed successfully.'), 'type' => 'success']);
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
