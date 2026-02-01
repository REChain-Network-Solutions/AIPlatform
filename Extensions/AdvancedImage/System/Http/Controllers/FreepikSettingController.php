<?php

namespace App\Extensions\AdvancedImage\System\Http\Controllers;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class FreepikSettingController extends Controller
{
    public function index(): View
    {
        return view('advanced-image::freepik-setting');
    }

    public function update(Request $request): RedirectResponse
    {
        $request->validate([
            'freepik_api_key' => 'required|string',
        ]);

        if (Helper::appIsNotDemo()) {
            setting([
                'freepik_api_key' => $request->get('freepik_api_key'),
            ])->save();
        }

        return back()
            ->with([
                'type'    => 'success',
                'message' => __('Freepik API key has been updated successfully.'),
            ]);
    }
}
