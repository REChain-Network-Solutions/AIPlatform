<?php

namespace App\Extensions\CheckoutRegistration\System\Http\Controllers;

use App\Http\Controllers\Controller;
use App\Models\Plan;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class CheckoutSettingsController extends Controller
{
    public function index()
    {
        $plans = Plan::where('active', 1)->get();

        return view('checkout-registration::settings.index', compact('plans'));
    }

    public function store(Request $request): RedirectResponse
    {
        $request->validate([
            'checkout_registration_status'  => 'sometimes|nullable|in:active,passive',
            'default_checkout_gateway'      => 'sometimes|nullable|in:stripe,paypal',
            'default_checkout_plan_id'      => 'sometimes|nullable|exists:plans,id',
        ]);

        setting([
            'checkout_registration_status' => $request->get('checkout_registration_status'),
            'default_checkout_gateway'     => $request->get('default_checkout_gateway'),
            'default_checkout_plan_id'     => $request->get('default_checkout_plan_id'),
        ])->save();

        return back()->with(['message' => __('Settings saved successfully.'), 'type' => 'success']);
    }
}
