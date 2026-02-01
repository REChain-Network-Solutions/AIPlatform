<?php

namespace App\Extensions\Xero\System\Http\Controllers;

use App\Http\Controllers\Controller;
use App\Models\User;
use Dcblogdev\Xero\Facades\Xero;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class XeroSettingController extends Controller
{
    public function index(): RedirectResponse|View
    {
        return view('xero::index');
    }

    public function update(Request $request): RedirectResponse
    {
        $XERO_CLIENT_ID = $request->input('XERO_CLIENT_ID');
        $XERO_CLIENT_SECRET = $request->input('XERO_CLIENT_SECRET');
        $XERO_REDIRECT_URI = $request->input('XERO_REDIRECT_URI');
        $XERO_LANDING_URL = $request->input('XERO_LANDING_URL');

        setting([
            'XERO_CLIENT_ID'     => $XERO_CLIENT_ID,
            'XERO_CLIENT_SECRET' => $XERO_CLIENT_SECRET,
            'XERO_REDIRECT_URI'  => $XERO_REDIRECT_URI,
            'XERO_LANDING_URL'   => $XERO_LANDING_URL,
        ])->save();

        return back()->with(['message', __('Xero settings updated successfully'), 'type' => 'success']);
    }

    public function createContacts(): RedirectResponse
    {
        ini_set('max_execution_time', 9999);
        ini_set('memory_limit', -1);

        // if required then use client id, client secret, redirect uri and landing url
        config([
            'xero.clientId'     => setting('XERO_CLIENT_ID'),
            'xero.clientSecret' => setting('XERO_CLIENT_SECRET'),
            'xero.redirectUri'  => setting('XERO_REDIRECT_URI'),
            'xero.landingUri'   => setting('XERO_LANDING_URL'),
        ]);

        if (! Xero::isConnected()) {
            return redirect('xero/connect');
        }

        $users = User::get();
        foreach ($users as $user) {
            if (empty($user->xero_account_id)) {
                $response = Xero::contacts()->store([
                    'Name' => $user->name,
                ]);
                $user->xero_account_id = $response['ContactID'] ?? null;
                $user->save();
            }
        }

        return back()->with([
            'type'    => 'success',
            'message' => trans('Contacts created successfully.'),
        ]);
    }
}
