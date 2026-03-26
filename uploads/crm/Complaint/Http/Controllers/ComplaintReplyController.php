<?php

namespace Modules\Complaint\Http\Controllers;

use App\Helper\Files;
use App\Helper\Reply;
use App\Http\Controllers\AccountBaseController;
use Modules\Complaint\Entities\ComplaintFile;
use Modules\Complaint\Entities\ComplaintReply;

class ComplaintReplyController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'app.menu.tickets';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('complaint', $this->user->modules));
            return $next($request);
        });
    }

    public function destroy($id)
    {
        $complaintReply = ComplaintReply::findOrFail($id);

        $this->deletePermission = user()->permission('delete_tickets');

        abort_403(!(
            $this->deletePermission == 'all'
            || ($this->deletePermission == 'added' && user()->id == $complaintReply->user_id)
            || ($this->deletePermission == 'owned' && (user()->id == $complaintReply->agent_id || user()->id == $complaintReply->user_id))
            || ($this->deletePermission == 'both' && (user()->id == $complaintReply->agent_id || user()->id == $complaintReply->added_by || user()->id == $complaintReply->user_id))
        ));


        $complaintFiles = ComplaintFile::where('complaint_reply_id', $id)->get();

        foreach ($complaintFiles as $file) {
            Files::deleteFile($file->hashname, 'complaint-files/' . $file->complaint_reply_id);
            $file->delete();
        }

        ComplaintReply::destroy($id);
        return Reply::success(__('messages.deleteSuccess'));

    }

}
