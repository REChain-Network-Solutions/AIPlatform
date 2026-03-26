<?php

namespace Modules\Feedback\Http\Controllers;

use App\Helper\Reply;
use App\Http\Controllers\AccountBaseController;
use Modules\Feedback\Http\Requests\StoreFeedbackChannel;
use Modules\Feedback\Http\Requests\UpdateFeedbackChannel;
use Modules\Feedback\Entities\FeedbackChannel;

class FeedbackChannelController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'feedback::modules.feedbackChannels';
        $this->activeSettingMenu = 'ticket_channels';
    }

    /**
     * @return \Illuminate\Contracts\Foundation\Application|\Illuminate\Contracts\View\Factory|\Illuminate\Contracts\View\View
     */
    public function create()
    {
        return view('feedback::feedback-settings.create-ticket-channel-modal');
    }

    /**
     * @param StoreFeedbackChannel $request
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function store(StoreFeedbackChannel $request)
    {
        $channel = new FeedbackChannel();
        $channel->channel_name = $request->channel_name;
        $channel->save();

        $allChannels = FeedbackChannel::all();

        $select = '<option value="">--</option>';

        foreach ($allChannels as $channel) {
            $select .= '<option value="' . $channel->id . '">' . mb_ucwords($channel->channel_name) . '</option>';
        }

        return Reply::successWithData(__('messages.recordSaved'), ['optionData' => $select]);
    }

    /**
     * @param int $id
     * @return \Illuminate\Contracts\Foundation\Application|\Illuminate\Contracts\View\Factory|\Illuminate\Contracts\View\View
     */
    public function edit($id)
    {
        $this->channel = FeedbackChannel::findOrFail($id);
        return view('feedback::feedback-settings.edit-ticket-channel-modal', $this->data);
    }

    /**
     * @param UpdateFeedbackChannel $request
     * @param int $id
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function update(UpdateFeedbackChannel $request, $id)
    {
        $channel = FeedbackChannel::findOrFail($id);
        $channel->channel_name = $request->channel_name;
        $channel->save();

        return Reply::success(__('messages.updateSuccess'));
    }

    /**
     * @param int $id
     * @return array
     */
    public function destroy($id)
    {
        FeedbackChannel::destroy($id);
        return Reply::success(__('messages.deleteSuccess'));
    }

    public function createModal()
    {
        return view('feedback::feedback-settings.channels.create-modal');
    }

}
