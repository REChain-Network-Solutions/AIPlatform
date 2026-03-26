<?php

namespace Modules\Complaint\Http\Controllers;

use App\Helper\Reply;
use App\Http\Controllers\AccountBaseController;
use Modules\Complaint\Http\Requests\StoreComplaintChannel;
use Modules\Complaint\Http\Requests\UpdateComplaintChannel;
use Modules\Complaint\Entities\ComplaintChannel;

class ComplaintChannelController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'complaint::modules.complaintChannels';
        $this->activeSettingMenu = 'ticket_channels';
    }

    /**
     * @return \Illuminate\Contracts\Foundation\Application|\Illuminate\Contracts\View\Factory|\Illuminate\Contracts\View\View
     */
    public function create()
    {
        return view('complaint::complaint-settings.create-ticket-channel-modal');
    }

    /**
     * @param StoreComplaintChannel $request
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function store(StoreComplaintChannel $request)
    {
        $channel = new ComplaintChannel();
        $channel->channel_name = $request->channel_name;
        $channel->save();

        $allChannels = ComplaintChannel::all();

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
        $this->channel = ComplaintChannel::findOrFail($id);
        return view('complaint::complaint-settings.edit-ticket-channel-modal', $this->data);
    }

    /**
     * @param UpdateComplaintChannel $request
     * @param int $id
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function update(UpdateComplaintChannel $request, $id)
    {
        $channel = ComplaintChannel::findOrFail($id);
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
        ComplaintChannel::destroy($id);
        return Reply::success(__('messages.deleteSuccess'));
    }

    public function createModal()
    {
        return view('complaint::complaint-settings.channels.create-modal');
    }

}
