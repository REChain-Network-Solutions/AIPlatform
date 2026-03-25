<?php

namespace App\Http\Controllers\Crm;

use App\Http\Controllers\Controller;
use App\Models\Crm\CrmActivity;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\View\View;

class CrmActivityController extends Controller
{
    public function index(Request $request): View
    {
        $query = CrmActivity::forUser(Auth::id())
            ->with(['contact', 'company', 'deal'])
            ->latest('due_at');

        if ($type = $request->input('type')) {
            $query->where('type', $type);
        }

        if ($request->boolean('overdue')) {
            $query->overdue();
        } elseif ($request->boolean('pending')) {
            $query->pending();
        }

        $activities = $query->paginate(25)->withQueryString();

        return view('crm.activities.index', compact('activities'));
    }

    public function store(Request $request): RedirectResponse
    {
        $validated = $request->validate([
            'type'           => 'required|in:note,call,email,meeting,task,deadline',
            'subject'        => 'required|string|max:255',
            'body'           => 'nullable|string',
            'due_at'         => 'nullable|date',
            'crm_contact_id' => 'nullable|exists:crm_contacts,id',
            'crm_company_id' => 'nullable|exists:crm_companies,id',
            'crm_deal_id'    => 'nullable|exists:crm_deals,id',
        ]);

        CrmActivity::create(array_merge($validated, ['user_id' => Auth::id()]));

        return back()->with('success', __('Activity logged.'));
    }

    public function markDone(CrmActivity $activity): JsonResponse
    {
        abort_unless($activity->user_id === Auth::id(), 403);
        $activity->markDone();

        return response()->json(['success' => true]);
    }

    public function destroy(CrmActivity $activity): RedirectResponse
    {
        abort_unless($activity->user_id === Auth::id(), 403);
        $activity->delete();

        return back()->with('success', __('Activity deleted.'));
    }
}
