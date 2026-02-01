<?php

namespace App\Extensions\ProductPhotography\System\Http\Controllers;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Models\Setting;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class PebblelySettingController extends Controller
{
    public function index(): View
    {
        return view('product-photography::setting');
    }

    public function update(Request $request): RedirectResponse
    {
        $request->validate([
            'pebblely_key' => 'required|string',
        ]);

        if (Helper::appIsNotDemo()) {
            $setting = Setting::query()->firstOrFail();

            $setting->update([
                'pebblely_key' => $request->get('pebblely_key'),
            ]);
        }

        return back()
            ->with([
                'type'    => 'success',
                'message' => __('Pebblely API key has been updated successfully.'),
            ]);
    }
}
