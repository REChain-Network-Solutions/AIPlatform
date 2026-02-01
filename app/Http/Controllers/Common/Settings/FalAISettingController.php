<?php

declare(strict_types=1);

namespace App\Http\Controllers\Common\Settings;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class FalAISettingController extends Controller
{
    public function index(Request $request): View|RedirectResponse
    {
        if (Helper::appIsDemo()) {
            return to_route('dashboard.user.index')->with([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        return view('default.panel.admin.common.settings.fal-ai');
    }

    public function update(Request $request): RedirectResponse
    {
        $data = $request->validate([
            'fal_ai_api_secret'        => 'required|string',
            'fal_ai_default_model'     => 'required|string',
            'enabled_flux_pro_kontext' => 'nullable',
        ]);

        $data['enabled_flux_pro_kontext'] = $request->has('enabled_flux_pro_kontext');
        $data['enabled_flux_2_flex'] = $request->has('enabled_flux_2_flex');

        setting($data)->save();

        return back()->with([
            'type'    => 'success',
            'message' => trans('Settings updated successfully.'),
        ]);
    }
}
