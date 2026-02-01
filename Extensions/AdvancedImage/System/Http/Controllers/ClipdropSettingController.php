<?php

namespace App\Extensions\AdvancedImage\System\Http\Controllers;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class ClipdropSettingController extends Controller
{
    public function index(): View
    {
        return view('advanced-image::clipdrop-setting');
    }

    public function update(Request $request): RedirectResponse
    {
        $request->validate([
            'clipdrop_api_key' => 'required|string',
        ]);

        if (Helper::appIsNotDemo()) {
            setting([
                'clipdrop_api_key' => $request->get('clipdrop_api_key'),
            ])->save();
        }

        return back()
            ->with([
                'type'    => 'success',
                'message' => __('Clipdrop API key has been updated successfully.'),
            ]);
    }
}
