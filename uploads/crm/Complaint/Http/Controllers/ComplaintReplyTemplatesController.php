<?php

namespace Modules\Complaint\Http\Controllers;

use App\Helper\Reply;
use Illuminate\Http\Request;
use App\Http\Controllers\AccountBaseController;
use Modules\Complaint\Http\Requests\StoreTemplate;
use Modules\Complaint\Http\Requests\UpdateTemplate;
use Modules\Complaint\Entities\ComplaintReplyTemplate;

class ComplaintReplyTemplatesController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'complaint::modules.complaintTemplates';
        $this->activeSettingMenu = 'ticket_reply_templates';
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        return view('complaint::complaint-settings.create-ticket-reply-template-modal');
    }

    /**
     * @param StoreTemplate $request
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function store(StoreTemplate $request)
    {
        $template = new ComplaintReplyTemplate();
        $template->reply_heading = trim_editor($request->reply_heading);
        $template->reply_text = $request->reply_text;
        $template->save();

        return Reply::success(__('messages.recordSaved'));
    }

    /**
     * @param int $id
     * @return \Illuminate\Contracts\Foundation\Application|\Illuminate\Contracts\View\Factory|\Illuminate\Contracts\View\View
     */
    public function edit($id)
    {
        $this->template = ComplaintReplyTemplate::findOrFail($id);
        return view('complaint::complaint-settings.edit-ticket-reply-template-modal', $this->data);
    }

    /**
     * @param UpdateTemplate $request
     * @param int $id
     * @return array
     * @throws \Froiden\RestAPI\Exceptions\RelatedResourceNotFoundException
     */
    public function update(UpdateTemplate $request, $id)
    {
        $template = ComplaintReplyTemplate::findOrFail($id);
        $template->reply_heading = $request->reply_heading;
        $template->reply_text = $request->reply_text;
        $template->save();

        return Reply::success(__('messages.templateUpdateSuccess'));
    }

    /**
     * @param int $id
     * @return array
     */
    public function destroy($id)
    {
        ComplaintReplyTemplate::destroy($id);
        return Reply::success(__('messages.deleteSuccess'));
    }

    public function fetchTemplate(Request $request)
    {
        $templateId = $request->templateId;
        $template = ComplaintReplyTemplate::findOrFail($templateId);
        return Reply::dataOnly(['replyText' => $template->reply_text, 'status' => 'success']);
    }

}
