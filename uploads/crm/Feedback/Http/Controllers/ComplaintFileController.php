<?php

namespace Modules\Feedback\Http\Controllers;

use App\Helper\Files;
use App\Helper\Reply;
use Illuminate\Http\Request;
use App\Http\Controllers\AccountBaseController;
use Modules\Feedback\Entities\FeedbackFile;
use Modules\Feedback\Entities\FeedbackReply;

class FeedbackFileController extends AccountBaseController
{

    public function store(Request $request)
    {
        if ($request->hasFile('file')) {
            $replyId = $request->feedback_reply_id;

            if ($request->feedback_reply_id == '') {
                $reply = new FeedbackReply();
                $reply->feedback_id = $request->feedback_id;
                $reply->user_id = $this->user->id; // Current logged in user
                $reply->save();
                $replyId = $reply->id;
            }

            foreach ($request->file as $fileData) {
                $file = new FeedbackFile();

                $file->feedback_reply_id = $replyId;

                $filename = Files::uploadLocalOrS3($fileData, FeedbackFile::FILE_PATH . '/' . $replyId);

                $file->user_id = $this->user->id;
                $file->filename = $fileData->getClientOriginalName();
                $file->hashname = $filename;
                $file->size = $fileData->getSize();
                $file->save();

            }
        }

        return Reply::dataOnly(['status' => 'success']);
    }

    /**
     * @param Request $request
     * @param int $id
     * @return array
     */
    public function destroy(Request $request, $id)
    {
        $file = FeedbackFile::findOrFail($id);

        Files::deleteFile($file->hashname, 'feedback-files/' . $file->feedback_reply_id);
        FeedbackFile::destroy($id);

        return Reply::success(__('messages.deleteSuccess'));
    }

    public function show($id)
    {
        $file = FeedbackFile::whereRaw('md5(id) = ?', $id)->firstOrFail();
        $this->filepath = $file->file_url;
        return view('tasks.files.view', $this->data);
    }

    /**
     * @param mixed $id
     * @return \Symfony\Component\HttpFoundation\BinaryFileResponse|\Symfony\Component\HttpFoundation\StreamedResponse
     */
    public function download($id)
    {
        $file = FeedbackFile::whereRaw('md5(id) = ?', $id)->firstOrFail();
        return download_local_s3($file, 'feedback-files/' . $file->feedback_reply_id . '/' . $file->hashname);
    }

}
