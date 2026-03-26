<?php

namespace Modules\Complaint\Http\Controllers;

use App\Helper\Reply;
use App\Http\Controllers\AccountBaseController;
use Modules\Complaint\Http\Requests\StoreComplaintType;
use Modules\Complaint\Http\Requests\UpdateComplaintType;
use Modules\Complaint\Entities\ComplaintType;

class ComplaintTypeController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'complaint::modules.complaintTypes';
        $this->activeSettingMenu = 'ticket_types';
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        $this->ticketTypes = ComplaintType::all();
        return view('complaint::complaint-settings.create-ticket-type-modal', $this->data);
    }

    /**
     * @param StoreComplaintType $request
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function store(StoreComplaintType $request)
    {
        $type = new ComplaintType();
        $type->type = $request->type;
        $type->save();

        $allTypes = ComplaintType::all();

        $select = '';

        foreach($allTypes as $type){
            $select .= '<option value="'.$type->id.'">'.$type->type.'</option>';
        }

        return Reply::successWithData(__('messages.recordSaved'), ['optionData' => $select]);
    }

    /**
     * Show the form for editing the specified resource.
     *
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function edit($id)
    {
        $this->type = ComplaintType::findOrFail($id);
        return view('complaint::complaint-settings.edit-ticket-type-modal', $this->data);
    }

    /**
     * @param UpdateComplaintType $request
     * @param int $id
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function update(UpdateComplaintType $request, $id)
    {
        $type = ComplaintType::findOrFail($id);
        $type->type = $request->type;
        $type->save();

        return Reply::success(__('messages.updateSuccess'));
    }

    /**
     * @param int $id
     * @return array
     */
    public function destroy($id)
    {
        ComplaintType::destroy($id);

        return Reply::success(__('messages.deleteSuccess'));
    }

}
