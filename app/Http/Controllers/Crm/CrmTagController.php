<?php

namespace App\Http\Controllers\Crm;

use App\Http\Controllers\Controller;
use App\Models\Crm\CrmTag;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\View\View;

class CrmTagController extends Controller
{
    public function index(): View
    {
        $tags = CrmTag::where('user_id', Auth::id())
            ->withCount(['contacts', 'companies', 'deals'])
            ->orderBy('name')
            ->get();

        return view('crm.settings.tags', compact('tags'));
    }

    public function store(Request $request): RedirectResponse
    {
        $validated = $request->validate([
            'name'  => 'required|string|max:100',
            'color' => 'nullable|string|max:20',
        ]);

        CrmTag::create([
            'user_id' => Auth::id(),
            'name'    => $validated['name'],
            'color'   => $validated['color'] ?? '#6366f1',
        ]);

        return back()->with('success', __('Tag created.'));
    }

    public function update(Request $request, CrmTag $tag): RedirectResponse
    {
        abort_unless($tag->user_id === Auth::id(), 403);

        $validated = $request->validate([
            'name'  => 'required|string|max:100',
            'color' => 'nullable|string|max:20',
        ]);

        $tag->update($validated);

        return back()->with('success', __('Tag updated.'));
    }

    public function destroy(CrmTag $tag): RedirectResponse
    {
        abort_unless($tag->user_id === Auth::id(), 403);
        $tag->taggables()->delete();
        $tag->delete();

        return back()->with('success', __('Tag deleted.'));
    }

    public function search(Request $request): JsonResponse
    {
        $tags = CrmTag::where('user_id', Auth::id())
            ->when($request->input('q'), fn ($q, $s) => $q->where('name', 'like', "%{$s}%"))
            ->orderBy('name')
            ->limit(20)
            ->get(['id', 'name', 'color']);

        return response()->json($tags);
    }
}
