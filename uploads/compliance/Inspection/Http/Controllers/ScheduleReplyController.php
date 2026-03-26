<?php
namespace Modules\Inspection\Http\Controllers;

use App\Helper\Files;
use App\Helper\Reply;
use App\Http\Controllers\AccountBaseController;
use Modules\Inspection\Entities\ScheduleFile;
use Modules\Inspection\Entities\ScheduleReply;

class ScheduleReplyController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'app.menu.tickets';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('inspections', $this->user->modules));
            return $next($request);
        });
    }

    public function destroy($id)
    {
        $scheduleReply = ScheduleReply::findOrFail($id);
        $this->deletePermission = user()->permission('delete_inspection');


        $scheduleFiles = ScheduleFile::where('schedule_reply_id', $id)->get();

        foreach ($scheduleFiles as $file) {
            Files::deleteFile($file->hashname, 'schedule-files/' . $file->schedule_reply_id);
            $file->delete();
        }

        ScheduleReply::destroy($id);
        return Reply::success(__('messages.deleteSuccess'));

    }

}
