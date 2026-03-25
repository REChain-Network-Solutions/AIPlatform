<?php

namespace App\Http\Controllers\Crm;

use App\Http\Controllers\Controller;
use App\Models\Crm\CrmPipeline;
use App\Models\Crm\CrmPipelineStage;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\DB;
use Illuminate\View\View;

class CrmPipelineController extends Controller
{
    public function index(): View
    {
        $pipelines = CrmPipeline::forUser(Auth::id())
            ->with(['stages' => fn ($q) => $q->orderBy('order')])
            ->withCount('deals')
            ->orderBy('order')
            ->get();

        return view('crm.pipelines.index', compact('pipelines'));
    }

    public function create(): View
    {
        return view('crm.pipelines.create');
    }

    public function store(Request $request): RedirectResponse
    {
        $validated = $request->validate([
            'name'       => 'required|string|max:255',
            'currency'   => 'nullable|string|size:3',
            'is_default' => 'nullable|boolean',
            'stages'     => 'nullable|array',
            'stages.*.name'        => 'required|string|max:255',
            'stages.*.color'       => 'nullable|string|max:20',
            'stages.*.probability' => 'nullable|integer|min:0|max:100',
            'stages.*.is_won'      => 'nullable|boolean',
            'stages.*.is_lost'     => 'nullable|boolean',
        ]);

        DB::transaction(function () use ($validated) {
            $userId = Auth::id();

            if (!empty($validated['is_default'])) {
                CrmPipeline::forUser($userId)->update(['is_default' => false]);
            }

            $pipeline = CrmPipeline::create([
                'user_id'    => $userId,
                'name'       => $validated['name'],
                'currency'   => $validated['currency'] ?? 'USD',
                'is_default' => !empty($validated['is_default']),
                'order'      => CrmPipeline::forUser($userId)->count(),
            ]);

            foreach (($validated['stages'] ?? []) as $i => $stageData) {
                CrmPipelineStage::create([
                    'pipeline_id' => $pipeline->id,
                    'name'        => $stageData['name'],
                    'color'       => $stageData['color'] ?? '#6366f1',
                    'probability' => $stageData['probability'] ?? 0,
                    'order'       => $i,
                    'is_won'      => !empty($stageData['is_won']),
                    'is_lost'     => !empty($stageData['is_lost']),
                ]);
            }
        });

        return redirect()
            ->route('dashboard.crm.pipelines.index')
            ->with('success', __('Pipeline created successfully.'));
    }

    public function show(CrmPipeline $pipeline): View
    {
        $this->authorizePipeline($pipeline);
        $pipeline->load(['stages.deals']);

        return view('crm.pipelines.show', compact('pipeline'));
    }

    public function edit(CrmPipeline $pipeline): View
    {
        $this->authorizePipeline($pipeline);
        $pipeline->load(['stages' => fn ($q) => $q->orderBy('order')]);

        return view('crm.pipelines.edit', compact('pipeline'));
    }

    public function update(Request $request, CrmPipeline $pipeline): RedirectResponse
    {
        $this->authorizePipeline($pipeline);

        $validated = $request->validate([
            'name'       => 'required|string|max:255',
            'currency'   => 'nullable|string|size:3',
            'is_default' => 'nullable|boolean',
            'stages'     => 'nullable|array',
            'stages.*.id'          => 'nullable|integer',
            'stages.*.name'        => 'required|string|max:255',
            'stages.*.color'       => 'nullable|string|max:20',
            'stages.*.probability' => 'nullable|integer|min:0|max:100',
            'stages.*.is_won'      => 'nullable|boolean',
            'stages.*.is_lost'     => 'nullable|boolean',
            'stages.*.delete'      => 'nullable|boolean',
        ]);

        DB::transaction(function () use ($validated, $pipeline) {
            if (!empty($validated['is_default'])) {
                CrmPipeline::forUser(Auth::id())
                    ->where('id', '!=', $pipeline->id)
                    ->update(['is_default' => false]);
            }

            $pipeline->update([
                'name'       => $validated['name'],
                'currency'   => $validated['currency'] ?? $pipeline->currency,
                'is_default' => !empty($validated['is_default']),
            ]);

            $keptIds = [];
            foreach (($validated['stages'] ?? []) as $i => $stageData) {
                if (!empty($stageData['delete'])) {
                    if (!empty($stageData['id'])) {
                        $pipeline->stages()->whereKey($stageData['id'])->delete();
                    }
                    continue;
                }

                if (!empty($stageData['id'])) {
                    $stage = $pipeline->stages()->whereKey($stageData['id'])->first();
                    if ($stage) {
                        $stage->update([
                            'name'        => $stageData['name'],
                            'color'       => $stageData['color'] ?? $stage->color,
                            'probability' => $stageData['probability'] ?? 0,
                            'order'       => $i,
                            'is_won'      => !empty($stageData['is_won']),
                            'is_lost'     => !empty($stageData['is_lost']),
                        ]);
                        $keptIds[] = $stage->id;
                        continue;
                    }
                }

                $stage = CrmPipelineStage::create([
                    'pipeline_id' => $pipeline->id,
                    'name'        => $stageData['name'],
                    'color'       => $stageData['color'] ?? '#6366f1',
                    'probability' => $stageData['probability'] ?? 0,
                    'order'       => $i,
                    'is_won'      => !empty($stageData['is_won']),
                    'is_lost'     => !empty($stageData['is_lost']),
                ]);
                $keptIds[] = $stage->id;
            }
        });

        return redirect()
            ->route('dashboard.crm.pipelines.index')
            ->with('success', __('Pipeline updated successfully.'));
    }

    public function destroy(CrmPipeline $pipeline): RedirectResponse
    {
        $this->authorizePipeline($pipeline);
        abort_if($pipeline->deals()->exists(), 422, __('Cannot delete a pipeline that has deals.'));

        $pipeline->stages()->delete();
        $pipeline->delete();

        return redirect()
            ->route('dashboard.crm.pipelines.index')
            ->with('success', __('Pipeline deleted.'));
    }

    public function reorder(Request $request): \Illuminate\Http\JsonResponse
    {
        $validated = $request->validate(['ids' => 'required|array', 'ids.*' => 'integer']);
        $userId    = Auth::id();

        foreach ($validated['ids'] as $pos => $id) {
            CrmPipeline::forUser($userId)->where('id', $id)->update(['order' => $pos]);
        }

        return response()->json(['success' => true]);
    }

    private function authorizePipeline(CrmPipeline $pipeline): void
    {
        abort_unless($pipeline->user_id === Auth::id(), 403);
    }
}
