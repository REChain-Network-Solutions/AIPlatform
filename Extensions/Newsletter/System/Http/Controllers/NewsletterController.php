<?php

namespace App\Extensions\Newsletter\System\Http\Controllers;

use App\Extensions\Newsletter\System\Http\Requests\NewsletterRequest;
use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Models\EmailTemplates;

class NewsletterController extends Controller
{
    public function create()
    {
        return view('panel.email.form', [
            'action'   => route('dashboard.newsletter.store'),
            'method'   => 'POST',
            'template' => new EmailTemplates,
            'title'    => trans('Create Email Template'),
        ]);
    }

    public function store(NewsletterRequest $request)
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $data = $request->validated();

        EmailTemplates::query()->create($data);

        return redirect()
            ->route('dashboard.email-templates.index')
            ->with([
                'message' => 'Created Successfully', 'type' => 'success',
            ]);
    }
}
