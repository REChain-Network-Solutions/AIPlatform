<?php

namespace Modules\Feedback\Http\Controllers;

use App\Helper\Reply;
use App\Http\Controllers\AccountBaseController;
use Modules\Feedback\Http\Requests\StoreFeedbackType;
use Modules\Feedback\Http\Requests\UpdateFeedbackType;
use Modules\Feedback\Entities\FeedbackType;

class FeedbackTypeController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'feedback::modules.feedbackTypes';
        $this->activeSettingMenu = 'ticket_types';
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        $this->ticketTypes = FeedbackType::all();
        return view('feedback::feedback-settings.create-ticket-type-modal', $this->data);
    }

    /**
     * @param StoreFeedbackType $request
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function store(StoreFeedbackType $request)
    {
        $type = new FeedbackType();
        $type->type = $request->type;
        $type->save();

        $allTypes = FeedbackType::all();

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
        $this->type = FeedbackType::findOrFail($id);
        return view('feedback::feedback-settings.edit-ticket-type-modal', $this->data);
    }

    /**
     * @param UpdateFeedbackType $request
     * @param int $id
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function update(UpdateFeedbackType $request, $id)
    {
        $type = FeedbackType::findOrFail($id);
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
        FeedbackType::destroy($id);

        return Reply::success(__('messages.deleteSuccess'));
    }

}
