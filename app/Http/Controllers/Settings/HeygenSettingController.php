<?php

namespace App\Http\Controllers\Settings;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Http\Requests\Admin\Settings\UpdateHeygenSettingsRequest;
use App\Models\Setting;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;

class HeygenSettingController extends Controller
{
    public function index(): View
    {
        return view('default.panel.admin.settings.heygen');
    }

    public function update(UpdateHeygenSettingsRequest $request): RedirectResponse
    {
        if (Helper::appIsNotDemo()) {
            $setting = Setting::getCache();
            $setting->heygen_secret_key = $request->validated('heygen_secret_key');
            $setting->save();
        }

        return back()->with([
            'type'    => 'success',
            'message' => __('Heygen API key has been updated successfully.'),
        ]);
    }
}
