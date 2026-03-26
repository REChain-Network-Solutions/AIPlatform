<?php

namespace Modules\Feedback\Http\Controllers;

use App\Helper\Reply;
use App\Models\Company;
use Illuminate\Http\Request;
use App\Http\Controllers\AccountBaseController;
use Modules\Feedback\Entities\FeedbackCustomForm;

class FeedbackCustomFormController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'feedback::modules.feedbackForm';
        $this->middleware(function ($request, $next) {
            if (!in_array('feedback', $this->user->modules)) {
                abort(403);
            }

            return $next($request);
        });
    }

    public function index()
    {
        $this->feedbackFormFields = FeedbackCustomForm::get();

        return view('feedback::feedback.ticket-form.index', $this->data);
    }

    /**
     * update record
     *
     * @return \Illuminate\Http\Response
     */
    public function update(Request $request, $id)
    {
            FeedbackCustomForm::where('id', $id)->update([
                'status' => $request->status
            ]);

        return Reply::dataOnly([]);
    }

    /**
     * sort fields order
     *
     * @return \Illuminate\Http\Response
     */
    public function sortFields()
    {
        $sortedValues = request('sortedValues');

        foreach ($sortedValues as $key => $value) {
            FeedbackCustomForm::where('id', $value)->update(['field_order' => $key + 1]);
        }

        return Reply::dataOnly([]);
    }

}
