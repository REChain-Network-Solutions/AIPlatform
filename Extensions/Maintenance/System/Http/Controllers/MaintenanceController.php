<?php

namespace App\Extensions\Maintenance\System\Http\Controllers;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;

class MaintenanceController extends Controller
{
    public function index()
    {
        $maintenance = Cache::get('maintenance');

        return view('maintenance::index', [
            'maintenance' => $maintenance,
        ]);
    }

    public function update(Request $request)
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $request->validate([
            'image' => 'sometimes|image|mimes:jpeg,png,jpg,gif,svg,webp|max:2048',
        ]);

        $request->merge([
            'maintenance_mode' => $request->has('maintenance_mode'),
        ]);

        if ($image = $request->file('image')) {
            $data = $request->all();
            $data['image'] = $image->store('maintenance', 'public');
        } else {
            $maintenance = Cache::get('maintenance');

            $data = $request->except('image');

            if (isset($maintenance['image'])) {
                $data['image'] = $maintenance['image'];
            }
        }

        Cache::put('maintenance', $data);

        return back()->with([
            'type'    => 'success',
            'message' => 'Maintenance settings updated successfully.',
        ]);
    }
}
