<?php

namespace App\Extensions\Hubspot\System\Http\Controllers;

use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\Factory;
use Illuminate\Contracts\View\View;
use Illuminate\Foundation\Application;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class HubspotController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index(): Factory|Application|View|\Illuminate\Contracts\Foundation\Application
    {
        return view('hubspot::index');
    }

    public function store(Request $request): RedirectResponse
    {
        $request->validate(['hubspot_access_token' => 'required']);

        setting([
            'hubspot_crm_contact_register' => $request->has('hubspot_crm_contact_register'),
            'hubspot_access_token'         => $request->get('hubspot_access_token'),
        ])->save();

        return back()->with([
            'message' => __('Saved Successfully'), 'type' => 'success',
        ]);
    }
}
