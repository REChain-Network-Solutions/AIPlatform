<?php

namespace App\Extensions\Mailchimp\System\Http\Controllers;

use App\Extensions\Mailchimp\System\Http\Requests\MailchimpSettingRequest;
use App\Http\Controllers\Controller;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;

class MailchimpController extends Controller
{
    public function index(): View
    {
        return view('mailchimp::index');
    }

    public function store(MailchimpSettingRequest $request): RedirectResponse
    {
        $request->validate([
            'mailchimp_api_key' => 'required',
            'mailchimp_list_id' => 'required',
        ]);

        setting($request->validated())->save();

        return back()->with(['message' => __('Saved Successfully'), 'type' => 'success']);
    }
}
