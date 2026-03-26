<?php

namespace Modules\Complaint\Http\Controllers;

use App\Helper\Files;
use App\Helper\Reply;
use Illuminate\Http\Request;
use App\Http\Controllers\AccountBaseController;
use Modules\Complaint\Entities\ComplaintFile;
use Modules\Complaint\Entities\ComplaintReply;

class ComplaintFileController extends AccountBaseController
{

    public function store(Request $request)
    {
        if ($request->hasFile('file')) {
            $replyId = $request->complaint_reply_id;

            if ($request->complaint_reply_id == '') {
                $reply = new ComplaintReply();
                $reply->complaint_id = $request->complaint_id;
                $reply->user_id = $this->user->id; // Current logged in user
                $reply->save();
                $replyId = $reply->id;
            }

            foreach ($request->file as $fileData) {
                $file = new ComplaintFile();

                $file->complaint_reply_id = $replyId;

                $filename = Files::uploadLocalOrS3($fileData, ComplaintFile::FILE_PATH . '/' . $replyId);

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
        $file = ComplaintFile::findOrFail($id);

        Files::deleteFile($file->hashname, 'complaint-files/' . $file->complaint_reply_id);
        ComplaintFile::destroy($id);

        return Reply::success(__('messages.deleteSuccess'));
    }

    public function show($id)
    {
        $file = ComplaintFile::whereRaw('md5(id) = ?', $id)->firstOrFail();
        $this->filepath = $file->file_url;
        return view('tasks.files.view', $this->data);
    }

    /**
     * @param mixed $id
     * @return \Symfony\Component\HttpFoundation\BinaryFileResponse|\Symfony\Component\HttpFoundation\StreamedResponse
     */
    public function download($id)
    {
        $file = ComplaintFile::whereRaw('md5(id) = ?', $id)->firstOrFail();
        return download_local_s3($file, 'complaint-files/' . $file->complaint_reply_id . '/' . $file->hashname);
    }

}
