<?php

namespace App\Extensions\Announcement\System\Http\Controllers;

use App\Extensions\Announcement\System\Http\Requests\AnnouncementRequest;
use App\Extensions\Announcement\System\Models\Announcement;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Support\Facades\Cache;

class AnnouncementController extends Controller
{
    // public announcement create
    public function create(): View
    {
        return view('announcement::create', [
            'action' => route('dashboard.admin.public-announcement.store'),
        ]);
    }

    // store
    public function store(AnnouncementRequest $request): JsonResponse
    {
        $entryLength = count($request->title);
        for ($i = 0; $i < $entryLength; $i++) {
            Announcement::create([
                'title' 	 => $request->title[$i],
                'type' 		 => $request->type[$i],
                'active' 	=> filter_var($request->active[$i], FILTER_VALIDATE_BOOLEAN),
            ]);
        }

        $this->cacheRefresh();

        return response()->json(['message' => __('Saved Successfully'), 'type' => 'success']);
    }

    // edit
    public function edit(Announcement $announcement): View
    {
        return view('announcement::edit', [
            'announcement' => $announcement,
            'action'       => route('dashboard.admin.public-announcement.update', $announcement->id),
        ]);
    }

    // update
    public function update(Announcement $announcement, AnnouncementRequest $request): JsonResponse
    {
        $announcement->update([
            'title' 	 => $request->title,
            'type' 		 => $request->type,
            'active' 	=> filter_var($request->active, FILTER_VALIDATE_BOOLEAN),
        ]);

        $this->cacheRefresh();

        return response()->json(['message' => __('Saved Successfully'), 'type' => 'success']);
    }

    // delete
    public function destroy(Announcement $announcement): RedirectResponse
    {
        $announcement->delete();
        $this->cacheRefresh();

        return back()->with(['message' => __('Deleted Successfully'), 'type' => 'success']);
    }

    // cache refresh
    protected function cacheRefresh()
    {
        Cache::forget('announcements');
        Cache::forget('public_announcements');
    }
}
