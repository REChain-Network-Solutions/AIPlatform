<?php

namespace Modules\Feedback\Http\Controllers;

use App\Helper\Reply;
use App\Models\BaseModel;
use App\Http\Controllers\AccountBaseController;
use Modules\Feedback\Http\Requests\StoreFeedbackGroup;
use Modules\Feedback\Entities\FeedbackGroup;

class FeedbackGroupController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'feedback::modules.feedbackAgent';
        $this->activeSettingMenu = 'ticket_settings';
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        $this->groups = FeedbackGroup::all();
        return view('feedback::feedback-settings.group-modal', $this->data);
    }

    /**
     * @param StoreFeedbackGroup $request
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function store(StoreFeedbackGroup $request)
    {
        $group = new FeedbackGroup();
        $group->group_name = $request->group_name;
        $group->save();

        $groups = FeedbackGroup::all();
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
        FeedbackGroup::destroy($id);
        return Reply::success(__('messages.deleteSuccess'));
    }

}
