<?php

namespace App\Extensions\AIRealtimeImage\System\Http\Controllers;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class TogetherSettingController extends Controller
{
    public function index(): View
    {
        return view('ai-realtime-image::setting');
    }

    public function update(Request $request): RedirectResponse
    {
        $request->validate(['together_api_key' => 'required|string']);

        if (Helper::appIsNotDemo()) {
            setting([
                'together_api_key' => $request->together_api_key,
            ])->save();
        }

        return back()
            ->with([
                'type'    => 'success',
                'message' => __('Together API key has been updated successfully.'),
            ]);
    }
}
