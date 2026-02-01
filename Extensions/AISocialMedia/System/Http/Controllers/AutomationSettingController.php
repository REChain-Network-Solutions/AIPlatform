<?php

namespace App\Extensions\AISocialMedia\System\Http\Controllers;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;

class AutomationSettingController extends Controller
{
    public function index()
    {
        return view('ai-social-media::settings');
    }

    public function update(Request $request)
    {
        $request->validate([
            'instagram_app_id'    => ['required', 'string', 'min:14', 'max:16'],
            'instagram_app_secret'=> ['required', 'string', 'min:30', 'max:32'],
        ]);

        setting($request->only('instagram_app_id', 'instagram_app_secret'))->save();

        return redirect()->back()->with([
            'type'    => 'success',
            'message' => __('Instagram settings updated successfully'),
        ]);
    }
}
