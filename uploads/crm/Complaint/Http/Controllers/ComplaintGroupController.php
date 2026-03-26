<?php

namespace Modules\Complaint\Http\Controllers;

use App\Helper\Reply;
use App\Models\BaseModel;
use App\Http\Controllers\AccountBaseController;
use Modules\Complaint\Http\Requests\StoreComplaintGroup;
use Modules\Complaint\Entities\ComplaintGroup;

class ComplaintGroupController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'complaint::modules.complaintAgent';
        $this->activeSettingMenu = 'ticket_settings';
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        $this->groups = ComplaintGroup::all();
        return view('complaint::complaint-settings.group-modal', $this->data);
    }

    /**
     * @param StoreComplaintGroup $request
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function store(StoreComplaintGroup $request)
    {
        $group = new ComplaintGroup();
        $group->group_name = $request->group_name;
        $group->save();

        $groups = ComplaintGroup::all();
        $options = BaseModel::options($groups, null, 'group_name');

        return Reply::successWithData(__('messages.recordSaved'), ['data' => $options]);
    }

    /**
     * Remove the specified resource from storage.
     *
     * @param  int  $id
     * @return \Illuminate\Http\Response
     */
    public function destroy($id)
    {
        ComplaintGroup::destroy($id);
        return Reply::success(__('messages.deleteSuccess'));
    }

}
