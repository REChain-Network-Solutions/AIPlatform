<?php

namespace Modules\Complaint\Http\Controllers;

use App\Helper\Reply;
use Illuminate\Http\Request;
use App\Http\Controllers\AccountBaseController;
use Modules\Complaint\Http\Requests\UpdateRequestComplaintEmailSetting;
use Modules\Complaint\Entities\ComplaintEmailSetting;

class ComplaintEmailSettingController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'app.menu.emailSync';
        $this->activeSettingMenu = 'ticket_settings';
    }

    public function update(UpdateRequestComplaintEmailSetting $request, $id)
    {
        $emailSetting = ComplaintEmailSetting::findOrFail($id);
        $data = $request->all();

        if ($request->has('status')) {
            $data['status'] = 1;

        } else {
            $data['status'] = 0;
        }

        $emailSetting->update($data);
        return Reply::success(__('messages.updateSuccess'));

    }

}
