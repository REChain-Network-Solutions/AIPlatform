<?php

namespace App\Http\Controllers\Crm;

use App\Http\Controllers\Controller;
use App\Models\Crm\CrmLeadSource;
use App\Models\Crm\CrmLeadStatus;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\View\View;

class CrmLeadSettingsController extends Controller
{
    // ─── Statuses ───────────────────────────────────────────────────────────

    public function statusesIndex(): View
    {
        $statuses = CrmLeadStatus::forUser(Auth::id())->get();

        return view('crm.settings.statuses', compact('statuses'));
    }

    public function statusStore(Request $request): RedirectResponse
    {
        $validated = $request->validate([
            'name'       => 'required|string|max:100',
            'color'      => 'nullable|string|max:20',
            'is_default' => 'nullable|boolean',
            'is_final'   => 'nullable|boolean',
        ]);

        $userId = Auth::id();

        if (!empty($validated['is_default'])) {
            CrmLeadStatus::forUser($userId)->update(['is_default' => false]);
        }

        CrmLeadStatus::create([
            'user_id'    => $userId,
            'name'       => $validated['name'],
            'color'      => $validated['color'] ?? '#6366f1',
            'position'   => CrmLeadStatus::forUser($userId)->count(),
            'is_default' => !empty($validated['is_default']),
            'is_final'   => !empty($validated['is_final']),
        ]);

        return back()->with('success', __('Status created.'));
    }

    public function statusUpdate(Request $request, CrmLeadStatus $status): RedirectResponse
    {
        $this->authorizeStatus($status);

        $validated = $request->validate([
            'name'       => 'required|string|max:100',
            'color'      => 'nullable|string|max:20',
            'is_default' => 'nullable|boolean',
            'is_final'   => 'nullable|boolean',
        ]);

        if (!empty($validated['is_default'])) {
            CrmLeadStatus::forUser(Auth::id())
                ->where('id', '!=', $status->id)
                ->update(['is_default' => false]);
        }

        $status->update([
            'name'       => $validated['name'],
            'color'      => $validated['color'] ?? $status->color,
            'is_default' => !empty($validated['is_default']),
            'is_final'   => !empty($validated['is_final']),
        ]);

        return back()->with('success', __('Status updated.'));
    }

    public function statusDestroy(CrmLeadStatus $status): RedirectResponse
    {
        $this->authorizeStatus($status);
        abort_if($status->leads()->exists(), 422, __('Cannot delete a status that has leads.'));

        $status->delete();

        return back()->with('success', __('Status deleted.'));
    }

    public function statusReorder(Request $request): JsonResponse
    {
        $validated = $request->validate(['ids' => 'required|array', 'ids.*' => 'integer']);
        $userId    = Auth::id();

        foreach ($validated['ids'] as $pos => $id) {
            CrmLeadStatus::where('id', $id)->where('user_id', $userId)->update(['position' => $pos]);
        }

        return response()->json(['success' => true]);
    }

    // ─── Sources ────────────────────────────────────────────────────────────

    public function sourcesIndex(): View
    {
        $sources = CrmLeadSource::where('user_id', Auth::id())->orderBy('name')->get();

        return view('crm.settings.sources', compact('sources'));
    }

    public function sourceStore(Request $request): RedirectResponse
    {
        $validated = $request->validate(['name' => 'required|string|max:100']);

        CrmLeadSource::create([
            'user_id'   => Auth::id(),
            'name'      => $validated['name'],
            'is_active' => true,
        ]);

        return back()->with('success', __('Source created.'));
    }

    public function sourceUpdate(Request $request, CrmLeadSource $source): RedirectResponse
    {
        $this->authorizeSource($source);

        $validated = $request->validate([
            'name'      => 'required|string|max:100',
            'is_active' => 'nullable|boolean',
        ]);

        $source->update([
            'name'      => $validated['name'],
            'is_active' => !empty($validated['is_active']),
        ]);

        return back()->with('success', __('Source updated.'));
    }

    public function sourceDestroy(CrmLeadSource $source): RedirectResponse
    {
        $this->authorizeSource($source);
        abort_if($source->leads()->exists(), 422, __('Cannot delete a source that has leads.'));

        $source->delete();

        return back()->with('success', __('Source deleted.'));
    }

    private function authorizeStatus(CrmLeadStatus $status): void
    {
        abort_unless($status->user_id === Auth::id(), 403);
    }

    private function authorizeSource(CrmLeadSource $source): void
    {
        abort_unless($source->user_id === Auth::id(), 403);
    }
}
