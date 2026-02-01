<?php

namespace App\Extensions\AdvancedImage\System\Http\Controllers;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class NovitaSettingController extends Controller
{
    public function index(): View
    {
        return view('advanced-image::novita-setting');
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
}
