<?php

namespace App\Http\Controllers\Admin\Config;

use App\Http\Controllers\Controller;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class PremiumAdvantagesController extends Controller
{
    public function index()
    {
        return view('default.panel.admin.config.premium-advantages', [
            'premiumAdvantages' => [
                'premium_advantages_1_label' => setting('premium_advantages_1_label', 'Unlimited Credits'),
                'premium_advantages_2_label' => setting('premium_advantages_2_label', 'Access to All Templates'),
                'premium_advantages_3_label' => setting('premium_advantages_3_label', 'External Chatbots'),
                'premium_advantages_4_label' => setting('premium_advantages_4_label', 'o1-mini and DeepSeek R1'),
                'premium_advantages_5_label' => setting('premium_advantages_5_label', 'Premium Support'),
            ],
        ]);
    }

    public function store(Request $request): RedirectResponse
    {
        $data = $request->validate([
            'premium_advantages_1_label' => 'required|string|max:255',
            'premium_advantages_2_label' => 'required|string|max:255',
            'premium_advantages_3_label' => 'required|string|max:255',
            'premium_advantages_4_label' => 'required|string|max:255',
            'premium_advantages_5_label' => 'required|string|max:255',
        ]);

        setting($data)->save();

        return redirect()->back()->with([
            'message' => __('Premium advantages updated successfully.'),
            'type'    => 'success',
        ]);
    }
}
