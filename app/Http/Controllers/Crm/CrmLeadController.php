<?php

namespace App\Http\Controllers\Crm;

use App\Http\Controllers\Controller;
use App\Models\Crm\CrmContact;
use App\Models\Crm\CrmDeal;
use App\Models\Crm\CrmLead;
use App\Models\Crm\CrmLeadSource;
use App\Models\Crm\CrmLeadStatus;
use App\Models\Crm\CrmPipeline;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\DB;
use Illuminate\View\View;

class CrmLeadController extends Controller
{
    public function index(Request $request): View
    {
        $userId   = Auth::id();
        $statuses = CrmLeadStatus::forUser($userId)->with(['leads' => function ($q) use ($userId) {
            $q->forUser($userId)->open()->with('source')->orderBy('order');
        }])->get();

        // Seed default statuses if none exist
        if ($statuses->isEmpty()) {
            $this->seedDefaultStatuses($userId);
            $statuses = CrmLeadStatus::forUser($userId)->with(['leads' => function ($q) use ($userId) {
                $q->forUser($userId)->open()->with('source')->orderBy('order');
            }])->get();
        }

        $sources = CrmLeadSource::forUser($userId)->get();

        // List view with filters
        $query = CrmLead::forUser($userId)->with(['status', 'source'])->latest();

        if ($search = $request->input('search')) {
            $query->where(function ($q) use ($search) {
                $q->where('title', 'like', "%{$search}%")
                    ->orWhere('contact_name', 'like', "%{$search}%")
                    ->orWhere('contact_email', 'like', "%{$search}%")
                    ->orWhere('company_name', 'like', "%{$search}%");
            });
        }

        if ($statusId = $request->input('status_id')) {
            $query->where('crm_lead_status_id', $statusId);
        }

        if ($sourceId = $request->input('source_id')) {
            $query->where('crm_lead_source_id', $sourceId);
        }

        $leads = $query->paginate(20)->withQueryString();

        $view = $request->input('view', 'kanban'); // kanban or list

        return view('crm.leads.index', compact('statuses', 'sources', 'leads', 'view'));
    }

    public function create(): View
    {
        $userId  = Auth::id();
        $statuses = CrmLeadStatus::forUser($userId)->get();
        $sources  = CrmLeadSource::forUser($userId)->get();

        return view('crm.leads.create', compact('statuses', 'sources'));
    }

    public function store(Request $request): RedirectResponse
    {
        $validated = $request->validate([
            'title'               => 'required|string|max:255',
            'contact_name'        => 'nullable|string|max:255',
            'contact_email'       => 'nullable|email|max:255',
            'contact_phone'       => 'nullable|string|max:30',
            'company_name'        => 'nullable|string|max:255',
            'website'             => 'nullable|url|max:255',
            'value'               => 'nullable|numeric|min:0',
            'currency'            => 'nullable|string|size:3',
            'crm_lead_status_id'  => 'nullable|exists:crm_lead_statuses,id',
            'crm_lead_source_id'  => 'nullable|exists:crm_lead_sources,id',
            'description'         => 'nullable|string',
        ]);

        $lead = CrmLead::create(array_merge($validated, [
            'user_id'  => Auth::id(),
            'currency' => $validated['currency'] ?? 'USD',
            'value'    => $validated['value'] ?? 0,
        ]));

        return redirect()
            ->route('dashboard.crm.leads.show', $lead)
            ->with('success', __('Lead created successfully.'));
    }

    public function show(CrmLead $lead): View
    {
        $this->authorizeLead($lead);
        $lead->load(['status', 'source', 'assignedTo', 'convertedContact', 'convertedDeal']);

        return view('crm.leads.show', compact('lead'));
    }

    public function edit(CrmLead $lead): View
    {
        $this->authorizeLead($lead);
        $userId  = Auth::id();
        $statuses = CrmLeadStatus::forUser($userId)->get();
        $sources  = CrmLeadSource::forUser($userId)->get();

        return view('crm.leads.edit', compact('lead', 'statuses', 'sources'));
    }

    public function update(Request $request, CrmLead $lead): RedirectResponse
    {
        $this->authorizeLead($lead);

        $validated = $request->validate([
            'title'               => 'required|string|max:255',
            'contact_name'        => 'nullable|string|max:255',
            'contact_email'       => 'nullable|email|max:255',
            'contact_phone'       => 'nullable|string|max:30',
            'company_name'        => 'nullable|string|max:255',
            'website'             => 'nullable|url|max:255',
            'value'               => 'nullable|numeric|min:0',
            'currency'            => 'nullable|string|size:3',
            'crm_lead_status_id'  => 'nullable|exists:crm_lead_statuses,id',
            'crm_lead_source_id'  => 'nullable|exists:crm_lead_sources,id',
            'description'         => 'nullable|string',
        ]);

        $lead->update($validated);

        return redirect()
            ->route('dashboard.crm.leads.show', $lead)
            ->with('success', __('Lead updated successfully.'));
    }

    public function destroy(CrmLead $lead): RedirectResponse
    {
        $this->authorizeLead($lead);
        $lead->delete();

        return redirect()
            ->route('dashboard.crm.leads.index')
            ->with('success', __('Lead deleted.'));
    }

    /**
     * Move a lead to a different status column (kanban drag-drop).
     */
    public function moveStatus(Request $request, CrmLead $lead): JsonResponse
    {
        $this->authorizeLead($lead);

        $validated = $request->validate([
            'status_id' => 'required|exists:crm_lead_statuses,id',
            'order'     => 'nullable|integer|min:0',
        ]);

        $lead->update([
            'crm_lead_status_id' => $validated['status_id'],
            'order'              => $validated['order'] ?? 0,
        ]);

        return response()->json(['success' => true]);
    }

    /**
     * Convert lead → Contact (+ optionally a Deal).
     */
    public function convert(Request $request, CrmLead $lead): RedirectResponse
    {
        $this->authorizeLead($lead);
        abort_if($lead->is_converted, 422, 'Lead already converted.');

        $validated = $request->validate([
            'create_deal'  => 'nullable|boolean',
            'pipeline_id'  => 'nullable|exists:crm_pipelines,id',
            'stage_id'     => 'nullable|exists:crm_pipeline_stages,id',
        ]);

        DB::transaction(function () use ($lead, $validated) {
            // Create contact from lead
            $contact = CrmContact::create([
                'user_id'      => $lead->user_id,
                'first_name'   => $lead->contact_name ?? $lead->title,
                'email'        => $lead->contact_email,
                'phone'        => $lead->contact_phone,
                'company_name' => $lead->company_name,
                'website'      => $lead->website,
                'status'       => 'prospect',
            ]);

            $dealId = null;

            if (!empty($validated['create_deal']) && !empty($validated['pipeline_id']) && !empty($validated['stage_id'])) {
                $deal = CrmDeal::create([
                    'user_id'        => $lead->user_id,
                    'pipeline_id'    => $validated['pipeline_id'],
                    'stage_id'       => $validated['stage_id'],
                    'crm_contact_id' => $contact->id,
                    'title'          => $lead->title,
                    'value'          => $lead->value,
                    'currency'       => $lead->currency,
                    'description'    => $lead->description,
                ]);
                $dealId = $deal->id;
            }

            $lead->update([
                'converted_to_contact_id' => $contact->id,
                'converted_to_deal_id'    => $dealId,
                'converted_at'            => now(),
            ]);
        });

        return redirect()
            ->route('dashboard.crm.leads.show', $lead)
            ->with('success', __('Lead converted successfully.'));
    }

    private function authorizeLead(CrmLead $lead): void
    {
        abort_unless($lead->user_id === Auth::id(), 403);
    }

    private function seedDefaultStatuses(int $userId): void
    {
        $defaults = [
            ['name' => 'New',          'color' => '#6366f1', 'position' => 0, 'is_default' => true,  'is_final' => false],
            ['name' => 'Contacted',    'color' => '#3b82f6', 'position' => 1, 'is_default' => false, 'is_final' => false],
            ['name' => 'Qualified',    'color' => '#f59e0b', 'position' => 2, 'is_default' => false, 'is_final' => false],
            ['name' => 'Proposal Sent','color' => '#8b5cf6', 'position' => 3, 'is_default' => false, 'is_final' => false],
            ['name' => 'Won',          'color' => '#22c55e', 'position' => 4, 'is_default' => false, 'is_final' => true],
            ['name' => 'Lost',         'color' => '#ef4444', 'position' => 5, 'is_default' => false, 'is_final' => true],
        ];

        foreach ($defaults as $status) {
            CrmLeadStatus::create(array_merge($status, ['user_id' => $userId]));
        }
    }
}
