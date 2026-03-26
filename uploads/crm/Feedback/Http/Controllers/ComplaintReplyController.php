<?php

namespace Modules\Feedback\Http\Controllers;

use App\Helper\Files;
use App\Helper\Reply;
use App\Http\Controllers\AccountBaseController;
use Modules\Feedback\Entities\FeedbackFile;
use Modules\Feedback\Entities\FeedbackReply;

class FeedbackReplyController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'app.menu.tickets';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('feedback', $this->user->modules));
            return $next($request);
        });
    }

    public function destroy($id)
    {
        $feedbackReply = FeedbackReply::findOrFail($id);

        $this->deletePermission = user()->permission('delete_tickets');

        abort_403(!(
            $this->deletePermission == 'all'
            || ($this->deletePermission == 'added' && user()->id == $feedbackReply->user_id)
            || ($this->deletePermission == 'owned' && (user()->id == $feedbackReply->agent_id || user()->id == $feedbackReply->user_id))
            || ($this->deletePermission == 'both' && (user()->id == $feedbackReply->agent_id || user()->id == $feedbackReply->added_by || user()->id == $feedbackReply->user_id))
        ));


        $feedbackFiles = FeedbackFile::where('feedback_reply_id', $id)->get();

        foreach ($feedbackFiles as $file) {
            Files::deleteFile($file->hashname, 'feedback-files/' . $file->feedback_reply_id);
            $file->delete();
        }

        FeedbackReply::destroy($id);
        return Reply::success(__('messages.deleteSuccess'));

    }

}
