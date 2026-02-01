<?php

namespace App\Extensions\PhotoStudio\System\Http\Controllers;

use App\Extensions\PhotoStudio\System\Models\PhotoStudio;
use App\Extensions\PhotoStudio\System\Services\NovitaService;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class NovitaSettingController extends Controller
{
    public function index(): View
    {
        return view('photo-studio::novitaSetting');
    }

    public function update(Request $request): RedirectResponse
    {
        $request->validate([
            'novita_api_key' => 'required|string',
        ]);

        if (Helper::appIsNotDemo()) {
            setting([
                'novita_api_key' => $request->get('novita_api_key'),
            ])->save();
        }

        return back()
            ->with([
                'type'    => 'success',
                'message' => __('Novita API key has been updated successfully.'),
            ]);
    }

    public function checkImageStatus(Request $request): \Illuminate\Http\JsonResponse
    {
        $ids = PhotoStudio::query()->where('status', 'in_progress')->pluck('request_id')->toArray();

        if (empty($ids)) {
            return response()->json(['data' => []]);
        }

        $data = [];
        foreach ($ids as $id) {
            $service = new NovitaService;
            $entry = $service->checkStatus($id);

            if ($entry['task']['status'] === 'TASK_STATUS_SUCCEED') {
                $imageUrl = $entry['images'][0]['image_url'];

                PhotoStudio::query()->where('request_id', $id)->update([
                    'status' => 'completed',
                    'photo'  => $imageUrl,
                ]);

                $item = PhotoStudio::query()->where('request_id', $id)->first();

                $data[] = [
                    'divId' => 'video-' . $id,
                    'html'  => view('photo-studio::particles.item', [
                        'item' => $item,
                    ])->render(),
                ];
            }
        }

        return response()->json(['data' => $data]);
    }
}
