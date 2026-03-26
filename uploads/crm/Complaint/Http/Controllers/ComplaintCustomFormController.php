<?php

namespace Modules\Complaint\Http\Controllers;

use App\Helper\Reply;
use App\Models\Company;
use Illuminate\Http\Request;
use App\Http\Controllers\AccountBaseController;
use Modules\Complaint\Entities\ComplaintCustomForm;

class ComplaintCustomFormController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'complaint::modules.complaintForm';
        $this->middleware(function ($request, $next) {
            if (!in_array('complaint', $this->user->modules)) {
                abort(403);
            }

            return $next($request);
        });
    }

    public function index()
    {
        $this->complaintFormFields = ComplaintCustomForm::get();

        return view('complaint::complaint.ticket-form.index', $this->data);
    }

    /**
     * update record
     *
     * @return \Illuminate\Http\Response
     */
    public function update(Request $request, $id)
    {
            ComplaintCustomForm::where('id', $id)->update([
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
            ComplaintCustomForm::where('id', $value)->update(['field_order' => $key + 1]);
        }

        return Reply::dataOnly([]);
    }

}
