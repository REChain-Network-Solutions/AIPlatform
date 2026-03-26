<?php

namespace Modules\Feedback\Http\Controllers;

use App\Helper\Reply;
use Illuminate\Http\Request;
use App\Http\Controllers\AccountBaseController;
use Modules\Feedback\Http\Requests\UpdateRequestFeedbackEmailSetting;
use Modules\Feedback\Entities\FeedbackEmailSetting;

class FeedbackEmailSettingController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'app.menu.emailSync';
        $this->activeSettingMenu = 'ticket_settings';
    }

    public function update(UpdateRequestFeedbackEmailSetting $request, $id)
    {
        $emailSetting = FeedbackEmailSetting::findOrFail($id);
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
