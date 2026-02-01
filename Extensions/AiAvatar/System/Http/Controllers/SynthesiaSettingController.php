<?php

namespace App\Extensions\AiAvatar\System\Http\Controllers;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Models\Setting;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class SynthesiaSettingController extends Controller
{
    public function index(): View
    {
        return view('ai-avatar::setting');
    }

    public function update(Request $request): RedirectResponse
    {
        $request->validate(['synthesia_secret_key' => 'required|string']);

        if (Helper::appIsNotDemo()) {
            $setting = Setting::query()->firstOrFail();
            $setting->synthesia_secret_key = $request->input('synthesia_secret_key');
            $setting->save();
        }

        return back()
            ->with([
                'type'    => 'success',
                'message' => __('Clipdrop API key has been updated successfully.'),
            ]);
    }
}
