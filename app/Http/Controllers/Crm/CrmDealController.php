<?php

namespace App\Http\Controllers\Crm;

use App\Http\Controllers\Controller;
use App\Models\Crm\CrmCompany;
use App\Models\Crm\CrmContact;
use App\Models\Crm\CrmDeal;
use App\Models\Crm\CrmPipeline;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\View\View;

class CrmDealController extends Controller
{
    public function index(Request $request): View
    {
        $pipelines = CrmPipeline::forUser(Auth::id())->with('stages.deals.contact')->get();

        // Default to first pipeline
        $activePipeline = $pipelines->firstWhere('id', $request->input('pipeline'))
            ?? $pipelines->first();

        return view('crm.deals.index', compact('pipelines', 'activePipeline'));
    }

    public function create(Request $request): View
    {
        $pipelines = CrmPipeline::forUser(Auth::id())->with('stages')->get();
        $contacts  = CrmContact::forUser(Auth::id())->orderBy('first_name')->get();
        $companies = CrmCompany::forUser(Auth::id())->orderBy('name')->get();

        return view('crm.deals.create', compact('pipelines', 'contacts', 'companies'));
    }

    public function store(Request $request): RedirectResponse
    {
        $validated = $request->validate([
            'pipeline_id'         => 'required|exists:crm_pipelines,id',
            'stage_id'            => 'required|exists:crm_pipeline_stages,id',
            'crm_contact_id'      => 'nullable|exists:crm_contacts,id',
            'crm_company_id'      => 'nullable|exists:crm_companies,id',
            'title'               => 'required|string|max:255',
            'value'               => 'nullable|numeric|min:0',
            'currency'            => 'required|string|size:3',
            'probability'         => 'nullable|integer|min:0|max:100',
            'expected_close_date' => 'nullable|date',
            'description'         => 'nullable|string',
        ]);

        $deal = CrmDeal::create(array_merge($validated, [
            'user_id' => Auth::id(),
            'status'  => 'open',
        ]));

        return redirect()
            ->route('dashboard.crm.deals.show', $deal)
            ->with('success', __('Deal created successfully.'));
    }

    public function show(CrmDeal $deal): View
    {
        $this->authorizeDeal($deal);
        $deal->load(['pipeline', 'stage', 'contact', 'company', 'activities' => fn ($q) => $q->latest()->limit(30)]);

        return view('crm.deals.show', compact('deal'));
    }

    public function edit(CrmDeal $deal): View
    {
        $this->authorizeDeal($deal);
        $pipelines = CrmPipeline::forUser(Auth::id())->with('stages')->get();
        $contacts  = CrmContact::forUser(Auth::id())->orderBy('first_name')->get();
        $companies = CrmCompany::forUser(Auth::id())->orderBy('name')->get();

        return view('crm.deals.edit', compact('deal', 'pipelines', 'contacts', 'companies'));
    }

    public function update(Request $request, CrmDeal $deal): RedirectResponse
    {
        $this->authorizeDeal($deal);

        $validated = $request->validate([
            'pipeline_id'         => 'required|exists:crm_pipelines,id',
            'stage_id'            => 'required|exists:crm_pipeline_stages,id',
            'crm_contact_id'      => 'nullable|exists:crm_contacts,id',
            'crm_company_id'      => 'nullable|exists:crm_companies,id',
            'title'               => 'required|string|max:255',
            'value'               => 'nullable|numeric|min:0',
            'currency'            => 'required|string|size:3',
            'probability'         => 'nullable|integer|min:0|max:100',
            'expected_close_date' => 'nullable|date',
            'status'              => 'required|in:open,won,lost',
            'description'         => 'nullable|string',
            'lost_reason'         => 'nullable|string',
        ]);

        if ($validated['status'] !== 'open' && ! $deal->closed_at) {
            $validated['closed_at'] = now()->toDateString();
        }

        $deal->update($validated);

        return redirect()
            ->route('dashboard.crm.deals.show', $deal)
            ->with('success', __('Deal updated successfully.'));
    }

    public function destroy(CrmDeal $deal): RedirectResponse
    {
        $this->authorizeDeal($deal);
        $deal->delete();

        return redirect()
            ->route('dashboard.crm.deals.index')
            ->with('success', __('Deal deleted.'));
    }

    /**
     * AJAX: move a deal to a new stage (kanban drag-drop).
     */
    public function moveStage(Request $request, CrmDeal $deal): JsonResponse
    {
        $this->authorizeDeal($deal);

        $validated = $request->validate([
            'stage_id' => 'required|exists:crm_pipeline_stages,id',
            'order'    => 'nullable|integer|min:0',
        ]);

        $deal->update($validated);

        return response()->json(['success' => true]);
    }

    private function authorizeDeal(CrmDeal $deal): void
    {
        abort_unless($deal->user_id === Auth::id(), 403);
    }
}
