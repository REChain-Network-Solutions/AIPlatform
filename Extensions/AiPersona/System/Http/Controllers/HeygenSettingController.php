<?php

namespace App\Extensions\AiPersona\System\Http\Controllers;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Models\Setting;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class HeygenSettingController extends Controller
{
    public function index(): View
    {
        return view('ai-persona::setting');
    }

    public function update(Request $request): RedirectResponse
    {
        $request->validate(['heygen_secret_key' => 'required|string']);

        if (Helper::appIsNotDemo()) {
            $setting = Setting::query()->firstOrFail();
            $setting->heygen_secret_key = $request->input('heygen_secret_key');
            $setting->save();
        }

        return back()
            ->with([
                'type'    => 'success',
                'message' => __('Heygen API key has been updated successfully.'),
            ]);
    }
}
