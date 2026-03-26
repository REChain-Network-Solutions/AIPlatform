<?php

namespace Modules\Inspection\Http\Controllers;

use Exception;
use Carbon\Carbon;
use App\Models\User;
use App\Helper\Files;
use App\Helper\Reply;
use Illuminate\Http\Request;
use App\Models\CompanyAddress;
use Modules\Units\Entities\Floor;
use Modules\Units\Entities\Tower;
use Modules\Inspection\Entities\Schedule;
use App\Http\Controllers\AccountBaseController;
use Modules\Inspection\Entities\ScheduleItems;
use Modules\Inspection\Entities\RecurringSchedule;
use Modules\Inspection\Entities\RecurringScheduleItems;
use Modules\Inspection\Http\Requests\StoreRecurringSchedule;
use Modules\Inspection\DataTables\ScheduleRecurringDataTable;
use Modules\Inspection\Http\Requests\UpdateRecurringSchedule;
use Modules\Inspection\DataTables\RecurringSchedulesDataTable;

class RecurringScheduleController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'Recurring Schedules Inspection';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('inspections', $this->user->modules));

            return $next($request);
        });
    }

    /**
     * Display a listing of the resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function index(ScheduleRecurringDataTable $dataTable)
    {
        $viewPermission = user()->permission('view_inspection');
        abort_403(!in_array($viewPermission, ['all']));

        return $dataTable->render('inspection::recurring-inspection_schedules.index', $this->data);
    }

    /**
     * Show the form for creating a new resource.
     *
     * @return \Illuminate\Http\Response
     */
    public function create()
    {
        $this->floors = Floor::all();
        $this->towers = Tower::all();

        $this->addPermission = user()->permission('add_inspection');
        abort_403(!in_array($this->addPermission, ['all']));

        $this->pageTitle = __('app.add') . ' ' . __('app.scheduleRecurring');
        $this->zero = '';

        $this->view = 'inspection::recurring-inspection_schedules.ajax.create';

        if (request()->ajax()) {
            $html = view($this->view, $this->data)->render();
            return Reply::dataOnly(['status' => 'success', 'html' => $html, 'title' => $this->pageTitle]);
        }


        return view('inspection::recurring-inspection_schedules.create', $this->data);
    }

    /**
     * @param StoreRecurringSchedule $request
     * @return array
     */
    public function store(StoreRecurringSchedule $request)
    {
        $items = $request->input('item_name');

        foreach ($items as $itm) {
            if (is_null($itm)) {
                return Reply::error(__('messages.itemBlank'));
            }
        }


        $recurringSchedule = new RecurringSchedule();
        $recurringSchedule->subject = $request->subject;
        $recurringSchedule->tower_id = $request->tower_id;
        $recurringSchedule->floor_id = $request->floor_id;
        $recurringSchedule->lokasi = $request->lokasi;
        $recurringSchedule->shift = $request->shift;
        $recurringSchedule->awal = $request->awal;
        $recurringSchedule->akhir = $request->akhir;

        $recurringSchedule->issue_date = !is_null($request->issue_date) ? Carbon::createFromFormat($this->company->date_format, $request->issue_date)->format('Y-m-d') : Carbon::now()->format('Y-m-d');

        $recurringSchedule->rotation = $request->rotation;
        $recurringSchedule->billing_cycle = $request->billing_cycle > 0 ? $request->billing_cycle : null;
        $recurringSchedule->unlimited_recurring = $request->billing_cycle < 0 ? 1 : 0;
        $recurringSchedule->created_by = $this->user->id;

        $recurringSchedule->immediate_schedule = ($request->immediate_schedule) ? 1 : 0;
        $recurringSchedule->status = 'active';
        $recurringSchedule->save();

        if($request->immediate_schedule){

            $schedule = new Schedule();
            $schedule->schedule_recurring_id = $recurringSchedule->id;
            $schedule->company_id = $recurringSchedule->company_id;
            $schedule->issue_date = Carbon::now()->format('Y-m-d');
            $schedule->subject = $recurringSchedule->subject;
            $schedule->floor_id = $recurringSchedule->floor_id;
            $schedule->tower_id = $recurringSchedule->tower_id;
            $schedule->lokasi = $recurringSchedule->lokasi;
            $schedule->shift = $recurringSchedule->shift;
            $schedule->awal = $recurringSchedule->awal;
            $schedule->akhir = $recurringSchedule->akhir;
            $schedule->save();

            foreach ($recurringSchedule->items as $item) {

                $scheduleItem = ScheduleItems::create(
                    [
                        'schedule_id' => $schedule->id,
                        'item_name' => $item->item_name
                    ]
                );



            }
        }

        return Reply::redirect(route('recurring-inspection_schedules.index'), __('messages.recordSaved'));
    }

    /**
     * Display the specified resource.
     *
     * @param int $id
     * @return \Illuminate\Http\Response
     */
    public function show($id)
    {
        $this->schedule = RecurringSchedule::with('recurrings')->findOrFail($id);

        $items = RecurringScheduleItems::where('schedule_recurring_id', $this->schedule->id)
            ->get();

        $this->settings = $this->company;


        $tab = request('tab');

        switch ($tab) {
        case 'inspection_schedules':
                return $this->inspection_schedules($id);
        default:
                $this->view = 'inspection::recurring-inspection_schedules.ajax.overview';
                break;
        }

        if (request()->ajax()) {
            $html = view($this->view, $this->data)->render();
            return Reply::dataOnly(['status' => 'success', 'html' => $html, 'title' => $this->pageTitle]);
        }

        $this->activeTab = $tab ?: 'overview';

        return view('inspection::recurring-inspection_schedules.show', $this->data);
    }

    /**
     * @param int $id
     * @return \Illuminate\Contracts\Foundation\Application|\Illuminate\Contracts\View\Factory|\Illuminate\Contracts\View\View|void
     */
    public function edit($id)
    {
        $this->floors = Floor::all();
        $this->towers = Tower::all();

        $this->schedule = RecurringSchedule::with('recurrings')->findOrFail($id);

        $this->editPermission = user()->permission('edit_inspection');
        abort_403(!($this->editPermission == 'all' ));


        return view('inspection::recurring-inspection_schedules.edit', $this->data);
    }

    /**
     * Update the specified resource in storage.
     *
     * @param UpdateRecurringSchedule $request
     * @param int $id
     * @return \Illuminate\Http\Response
     */

    public function update(UpdateRecurringSchedule $request, $id)
    {
        $schedule = RecurringSchedule::findOrFail($id);

        if($request->schedule_count == 0)
        {
            $items = $request->item_name;
            $item_ids = $request->item_ids;


            foreach ($items as $itm) {
                if (is_null($itm)) {
                    return Reply::error(__('messages.itemBlank'));
                }
            }

            $schedule->subject = $request->subject;
            $schedule->tower_id = $request->tower_id;
            $schedule->floor_id = $request->floor_id;
            $schedule->lokasi = $request->lokasi;
            $schedule->shift = $request->shift;
            $schedule->awal = $request->awal;
            $schedule->akhir = $request->akhir;

            $schedule->issue_date = Carbon::createFromFormat($this->company->date_format, $request->issue_date)->format('Y-m-d');

            $schedule->rotation = $request->rotation;
            $schedule->billing_cycle = $request->billing_cycle > 0 ? $request->billing_cycle : null;
            $schedule->unlimited_recurring = $request->billing_cycle < 0 ? 1 : 0;
            $schedule->created_by = $this->user->id;

            if ($request->rotation == 'weekly' || $request->rotation == 'bi-weekly') {
                $schedule->day_of_week = $request->day_of_week;
            }
            elseif ($request->rotation == 'monthly' || $request->rotation == 'quarterly' || $request->rotation == 'half-yearly' || $request->rotation == 'annually') {
                $schedule->day_of_month = $request->day_of_month;
            }

            if (request()->has('status')) {
                $schedule->status = $request->status;
            }

            $schedule->save();

            if (!empty($request->item_name) && is_array($request->item_name)) {
                // Step1 - Delete all invoice items which are not avaialable
                if (!empty($item_ids)) {
                    RecurringScheduleItems::whereNotIn('id', $item_ids)->where('schedule_recurring_id', $schedule->id)->delete();
                }

                // Step2&3 - Find old invoices items, update it and check if images are newer or older
                foreach ($items as $key => $item) {
                    $schedule_item_id = isset($item_ids[$key]) ? $item_ids[$key] : 0;

                    try {
                        $scheduleItem = RecurringScheduleItems::findOrFail($schedule_item_id);
                    }
                    catch(Exception) {
                        $scheduleItem = new RecurringScheduleItems();
                    }

                    $scheduleItem->schedule_recurring_id = $id;
                    $scheduleItem->item_name = $item;

                    $scheduleItem->save();


                }
            }

        } else {


            if (request()->has('status')) {
                $schedule->status = $request->status;
            }

            $schedule->save();
        }

        return Reply::redirect(route('recurring-inspection_schedules.index'), __('messages.recordSaved'));
    }

    /**
     * Remove the specified resource from storage.
     *
     * @param int $id
     * @return \Illuminate\Http\Response
     */
    public function destroy($id)
    {
        $this->deletePermission = user()->permission('delete_inspection');

        $recurringSchedule = RecurringSchedule::findOrFail($id);
        abort_403(!($this->deletePermission == 'all'));

        RecurringSchedule::destroy($id);

        return Reply::success(__('messages.deleteSuccess'));
    }

    /**
     * @param Request $request
     * @return array
     */
    public function changeStatus(Request $request)
    {
        $scheduleId = $request->scheduleId;
        $status = $request->status;
        $schedule = RecurringSchedule::findOrFail($scheduleId);

        if ($schedule) {
            $schedule->status = $status;
            $schedule->save();
        }

        return Reply::success(__('messages.updateSuccess'));
    }

    /**
     * @param RecurringSchedulesDataTable $dataTable
     * @param int $id
     * @return mixed
     */
    public function recurringSchedules(RecurringSchedulesDataTable $dataTable, $id)
    {
        $this->schedule = RecurringSchedule::findOrFail($id);

        return $dataTable->render('recurring-inspection_schedules.recurring-inspection_schedules', $this->data);
    }

    public function applyQuickAction(Request $request)
    {
        switch ($request->action_type) {
        case 'delete':
                $this->deleteRecords($request);
                return Reply::success(__('messages.deleteSuccess'));
        default:
                return Reply::error(__('messages.selectAction'));
        }
    }

    protected function deleteRecords($request)
    {
        abort_403(user()->permission('delete_inspection') != 'all');

        $items = explode(',', $request->row_ids);

        foreach ($items as $id) {
            RecurringSchedule::destroy($id);
        }
    }

    public function inspection_schedules($recurringID)
    {
        $dataTable = new RecurringSchedulesDataTable;
        $viewPermission = user()->permission('view_inspection');
        abort_403(!in_array($viewPermission, ['all']));

        $this->recurringID = $recurringID;
        $tab = request('tab');
        $this->activeTab = $tab ?: 'overview';

        $this->view = 'inspection::recurring-inspection_schedules.ajax.inspection_schedules';

        return $dataTable->render('inspection::recurring-inspection_schedules.show', $this->data);
    }

}
