<?php

namespace Modules\Inspection\Http\Controllers;

use Client;
use Exception;
use Carbon\Carbon;
use App\Models\User;
use App\Helper\Reply;
use App\Models\Country;
use App\Models\TicketGroup;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Modules\Inspection\Entities\Schedule;
use App\Http\Controllers\AccountBaseController;
use Modules\Inspection\Entities\ScheduleReply;
use Modules\Inspection\Http\Requests\UpdateSchedule;
use Modules\Inspection\DataTables\ScheduleInspectionDataTable;
use Modules\Inspection\Entities\ScheduleItems;

class ScheduleInspectionController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'Inspection';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('inspections', $this->user->modules));

            return $next($request);
        });
    }

    public function index(ScheduleInspectionDataTable $dataTable)
    {
        $this->viewInspectionPermission = user()->permission('view_inspection');
        abort_403(!in_array($this->viewInspectionPermission, ['all']));
        $this->groups =  TicketGroup::all();
        return $dataTable->render('inspection::schedule-inspection.index', $this->data);
    }

    public function applyQuickAction(Request $request)
    {
        switch ($request->action_type) {
            case 'delete':
                $this->deleteRecords($request);

                return Reply::success(__('messages.deleteSuccess'));
            case 'change-status':
                $this->changeBulkStatus($request);

                return Reply::success(__('messages.updateSuccess'));
            default:
                return Reply::error(__('messages.selectAction'));
        }
    }

    protected function deleteRecords($request)
    {
        abort_403(user()->permission('delete_inspection') != 'all');

        Schedule::whereIn('id', explode(',', $request->row_ids))->delete();
    }

    protected function changeBulkStatus($request)
    {
        abort_403(user()->permission('edit_inspection') != 'all');

        Schedule::whereIn('id', explode(',', $request->row_ids))->update(['status' => $request->status]);
    }

    public function show($id)
    {
        $this->viewInspectionPermission = user()->permission('view_inspection');
        $this->schedule = Schedule::with('worker', 'items', 'reply', 'reply.files', 'reply.user')
            ->where('id', $id)->first();
        abort_if(!$this->schedule, 404);
        $this->pageTitle = __('Inspection') . '-' . $this->schedule->subject;

        abort_403(!($this->viewInspectionPermission == 'all'
        ));
        $this->groups = TicketGroup::with('enabledAgents', 'enabledAgents.user')->get();
        return view('inspection::schedule-inspection.edit', $this->data);
    }

    public function update(UpdateSchedule $request, $id)
    {

        $schedule = Schedule::findOrFail($id);
        $schedule->status = $request->status;
        $schedule->save();

        $message = trim_editor($request->message);

        if ($message != '') {
            $reply = new ScheduleReply();
            $reply->message = $request->message;
            $reply->items = $request->items;
            $reply->schedule_id = $schedule->id;
            $reply->user_id = $this->user->id; // Current logged in user
            $reply->save();
            return Reply::successWithData(__('messages.ticketReplySuccess'), ['reply_id' => $reply->id]);
        }

        return Reply::dataOnly(['status' => 'success']);
    }

    public function destroy($id)
    {
        $schedule = Schedule::findOrFail($id);

        $this->deleteInspectionPermission = user()->permission('delete_inspection');
        abort_403(!($this->deleteInspectionPermission == 'all'
        ));

        Schedule::destroy($id);

        return Reply::success(__('messages.deleteSuccess'));
    }

    public function updateOtherData(Request $request, $id)
    {
        $schedule = Schedule::findOrFail($id);
        $schedule->agent_id = $request->agent_id;
        $schedule->priority = $request->priority;
        $schedule->save();

        $items = request()->items_name;
        $item_ids = request()->item_ids;
        $count_items = array_count_values($items);
        $items_total = isset($count_items['1']) ? $count_items['1'] : 0;
        $jumlah = count($item_ids);

        if (!empty($item_ids)) {
            ScheduleItems::whereNotIn('id', $item_ids)->where('schedule_id', $id)->delete();
        }

        foreach ($items as $key => $item) {

            $item_id = isset($item_ids[$key]) ? $item_ids[$key] : 0;
            try {
                $invoiceItem = ScheduleItems::findOrFail($item_id);
            } catch (Exception) {
                $invoiceItem = new ScheduleItems();
            }

            $invoiceItem->check = $item ? 1 : 0;
            $invoiceItem->saveQuietly();
        }

        if ($items_total == $jumlah) {
            $schedule = Schedule::findOrFail($id);
            $schedule->status = 'resolved';
            $schedule->save();

        } else if ($items_total != $jumlah) {
            $schedule = Schedule::findOrFail($id);
            $schedule->status = 'open';
            $schedule->save();
        }

        return Reply::success(__('messages.updateSuccess'));
    }

    public function refreshCount(Request $request)
    {
        $viewInspectionPermission = user()->permission('view_inspection');

        $inspection_schedules = Schedule::with('agent');

        if (!is_null($request->startDate) && $request->startDate != '') {
            $startDate = Carbon::createFromFormat($this->company->date_format, $request->startDate)->toDateString();
            $inspection_schedules->where(DB::raw('DATE(`updated_at`)'), '>=', $startDate);
        }

        if (!is_null($request->endDate) && $request->endDate != '') {
            $endDate = Carbon::createFromFormat($this->company->date_format, $request->endDate)->toDateString();
            $inspection_schedules->where(DB::raw('DATE(`updated_at`)'), '<=', $endDate);
        }

        if (!is_null($request->agentId) && $request->agentId != 'all') {
            $inspection_schedules->where('agent_id', '=', $request->agentId);
        }

        if (!is_null($request->priority) && $request->priority != 'all') {
            $inspection_schedules->where('priority', '=', $request->priority);
        }


        $inspection_schedules = $inspection_schedules->get();

        $openTickets = $inspection_schedules->filter(function ($value, $key) {
            return $value->status == 'open';
        })->count();

        $pendingTickets = $inspection_schedules->filter(function ($value, $key) {
            return $value->status == 'pending';
        })->count();

        $resolvedTickets = $inspection_schedules->filter(function ($value, $key) {
            return $value->status == 'resolved';
        })->count();

        $closedTickets = $inspection_schedules->filter(function ($value, $key) {
            return $value->status == 'closed';
        })->count();

        $totalTickets = $inspection_schedules->count();

        $scheduleData = [
            'totalTickets' => $totalTickets,
            'closedTickets' => $closedTickets,
            'openTickets' => $openTickets,
            'pendingTickets' => $pendingTickets,
            'resolvedTickets' => $resolvedTickets
        ];

        return Reply::dataOnly($scheduleData);
    }

    public function changeStatus(Request $request)
    {
        $schedule = Schedule::find($request->scheduleId);
        $this->editInspectionPermission = user()->permission('edit_inspection');

        abort_403(!($this->editInspectionPermission == 'all'));

        $schedule->update(['status' => $request->status]);

        return Reply::successWithData(__('messages.updateSuccess'), ['status' => 'success']);
    }
}
