<?php

namespace Modules\Inspection\Http\Controllers;

use Exception;
use Carbon\Carbon;
use App\Models\User;
use App\Helper\Files;
use App\Helper\Reply;
use App\Models\Company;
use App\Scopes\ActiveScope;
use Illuminate\Http\Request;
use App\Models\CompanyAddress;
use App\Models\Event;
// Units module is optional in many deployments.
// If it's not installed, inspection schedules should still work.
use Illuminate\Support\Facades\App;
use Modules\Inspection\Entities\Schedule;
use App\Http\Controllers\AccountBaseController;
use Modules\Inspection\Entities\ScheduleItems;
use Modules\Inspection\Http\Requests\StoreSchedule;
use Modules\Inspection\Http\Requests\UpdateSchedule;
use Modules\Inspection\DataTables\SchedulesDataTable;

class ScheduleController extends AccountBaseController
{

    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'Schedules';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('inspections', $this->user->modules));

            return $next($request);
        });
    }

    public function index(SchedulesDataTable $dataTable)
    {
        $viewPermission = user()->permission('view_inspection');
        abort_403(!in_array($viewPermission, ['all']));

        return $dataTable->render('inspection::schedules.index', $this->data);
    }

    public function create()
    {
        $this->floors = class_exists('Modules\Units\Entities\Floor') ? \Modules\Units\Entities\Floor::all() : collect();
        $this->towers = class_exists('Modules\Units\Entities\Tower') ? \Modules\Units\Entities\Tower::all() : collect();

        $this->addPermission = user()->permission('add_inspection');

        abort_403(!in_array($this->addPermission, ['all']));

        $this->employees = User::allEmployees(null, true, ($this->addPermission == 'all' ? 'all' : null));

        if (request('schedule') != '') {
            $this->scheduleId = request('schedule');
            $this->type = 'schedule';
            $this->schedule = Schedule::with('items')->findOrFail($this->scheduleId);
        }

        $this->pageTitle = __('Add Schedule');

        $this->companyAddresses = CompanyAddress::all();

        $schedule = new Schedule();

        if (request()->ajax()) {
            $html = view('inspection::schedules.ajax.create', $this->data)->render();

            return Reply::dataOnly(['status' => 'success', 'html' => $html, 'title' => $this->pageTitle]);
        }

        $this->view = 'inspection::schedules.ajax.create';

        return view('inspection::schedules.create', $this->data);

    }

    public function store(StoreSchedule $request)
    {
        $redirectUrl = urldecode($request->redirect_url);

        if ($redirectUrl == '') {
            $redirectUrl = route('schedules.index');
        }

        $schedule = new Schedule();
        $schedule->subject = $request->subject;
        $schedule->tower_id = $request->tower_id;
        $schedule->floor_id = $request->floor_id;
        $schedule->lokasi = $request->lokasi;
        $schedule->shift = $request->shift;
        $schedule->awal = $request->awal;
        $schedule->akhir = $request->akhir;
        $schedule->worker_id = $request->worker_id;
        $schedule->issue_date = Carbon::createFromFormat($this->company->date_format, $request->issue_date)->format('Y-m-d');

        $schedule->save();

        // Sync to WorkSuite calendar
        $this->upsertCalendarEventForSchedule($schedule);


        foreach ($schedule->items as $item) {

            $scheduleItem = ScheduleItems::create(
                [
                    'schedule_id' => $schedule->id,
                    'item_name' => $item->item_name
                ]
            );
        }

        // Log search
        $this->logSearchEntry($schedule->id, $schedule->subject, 'inspection_schedules.show', 'schedule');

        return Reply::successWithData(__('messages.recordSaved'), ['redirectUrl' => $redirectUrl]);
    }

    public function edit($id)
    {
        $this->floors = class_exists('Modules\Units\Entities\Floor') ? \Modules\Units\Entities\Floor::all() : collect();
        $this->towers = class_exists('Modules\Units\Entities\Tower') ? \Modules\Units\Entities\Tower::all() : collect();

        $this->schedule = Schedule::findOrFail($id);
        $this->editPermission = user()->permission('edit_inspection');

        abort_403(!(
            $this->editPermission == 'all'
        ));

        $this->employees = User::allEmployees(null, true, ($this->editPermission == 'all' ? 'all' : null));

        $this->pageTitle = $this->schedule->schedule_recurring_id;

        $this->companyAddresses = CompanyAddress::all();

        if (request()->ajax()) {
            $html = view('inspection::schedules.ajax.edit', $this->data)->render();

            return Reply::dataOnly(['status' => 'success', 'html' => $html, 'title' => $this->pageTitle]);
        }

        $this->view = 'inspection::schedules.ajax.edit';

        return view('inspection::schedules.create', $this->data);

    }

    public function update(UpdateSchedule $request, $id)
    {
        $schedule = Schedule::findOrFail($id);

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
        $schedule->worker_id = $request->worker_id;
        $schedule->issue_date = Carbon::createFromFormat($this->company->date_format, $request->issue_date)->format('Y-m-d');

        if ($request->has('status')) {
            $schedule->status = $request->status;
        }

        $schedule->save();

        // Sync to WorkSuite calendar
        $this->upsertCalendarEventForSchedule($schedule);


        if (!empty($request->item_name) && is_array($request->item_name)) {
            // Step1 - Delete all invoice items which are not avaialable
            if (!empty($item_ids)) {
                ScheduleItems::whereNotIn('id', $item_ids)->where('schedule_id', $schedule->id)->delete();
            }

            // Step2&3 - Find old invoices items, update it and check if images are newer or older
            foreach ($items as $key => $item) {
                $schedule_item_id = isset($item_ids[$key]) ? $item_ids[$key] : 0;

                try {
                    $scheduleItem = ScheduleItems::findOrFail($schedule_item_id);
                }
                catch(Exception) {
                    $scheduleItem = new ScheduleItems();
                }

                $scheduleItem->schedule_id = $id;
                $scheduleItem->item_name = $item;

                $scheduleItem->save();

            }
        }
        $redirectUrl = route('schedules.index');
        return Reply::successWithData(__('inspection::messages.updateReceive'), ['redirectUrl' => $redirectUrl]);
    }

    public function show($id)
    {
        $this->schedule = Schedule::findOrFail($id);
        /* Used for cancel schedule condition */
        // $this->firstSchedule = Schedule::orderBy('id', 'remark')->first();

        $this->viewPermission = user()->permission('view_inspection');
        $this->addSchedulesPermission = user()->permission('add_inspection');

        abort_403(!(
            $this->viewPermission == 'all'

        ));

        // $this->firstSchedule = Schedule::orderBy('id', 'remark')->first();

        return view('inspection::schedules.show', $this->data);

    }
    /**
     * XXXXXXXXXXX
     *
     * @return \Illuminate\Http\Response
     */
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

    public function destroy($id)
    {
        $firstSchedule = Schedule::orderBy('id', 'desc')->first();
        $schedule = Schedule::findOrFail($id);
        $this->deletePermission = user()->permission('delete_inspection');
        abort_403(!(
            $this->deletePermission == 'all'

        ));

        if ($firstSchedule->id == $id) {


            Schedule::destroy($id);

            return Reply::success(__('messages.deleteSuccess'));

        }
    }

    
    /**
     * Create or update the WorkSuite calendar entry for an inspection schedule.
     * Uses `events.scheduled_service_id` to link back to `inspection_schedules.id`.
     */
    protected function upsertCalendarEventForSchedule(Schedule $schedule): void
    {
        // Some deployments may not have the Event model even if the table exists.
        if (!class_exists(\App\Models\Event::class)) {
            return;
        }

        $companyId = $schedule->company_id ?? (company() ? company()->id : null);
        $userId = $schedule->worker_id ?? (user() ? user()->id : null);

        // Build start/end datetimes
        $date = $schedule->issue_date ? Carbon::parse($schedule->issue_date)->format('Y-m-d') : Carbon::now()->format('Y-m-d');
        $startTime = $schedule->awal ? Carbon::parse($schedule->awal)->format('H:i:s') : '09:00:00';
        $endTime = $schedule->akhir ? Carbon::parse($schedule->akhir)->format('H:i:s') : null;

        $start = Carbon::parse($date . ' ' . $startTime);
        $end = $endTime ? Carbon::parse($date . ' ' . $endTime) : (clone $start)->addHour();

        if ($end->lessThanOrEqualTo($start)) {
            $end = (clone $start)->addHour();
        }

        $event = Event::where('scheduled_service_id', $schedule->id)
            ->where('event_type', 'inspection')
            ->when($companyId, fn($q) => $q->where('company_id', $companyId))
            ->first();

        if (!$event) {
            $event = new Event();
            $event->scheduled_service_id = $schedule->id;
            $event->event_type = 'inspection';
            $event->service_status = 'scheduled';
            $event->added_by = user() ? user()->id : null;
        }

        $event->company_id = $companyId;
        $event->user_id = $userId;
        $event->host = $userId;

        // Required non-null fields in this WorkSuite schema
        $event->event_name = $schedule->subject ?? 'Inspection';
        $event->label_color = '#F57F25';
        $event->where = $schedule->lokasi ?? '';
        $event->description = $schedule->remark ?? '';
        $event->note = ($schedule->shift ? ('Shift: ' . $schedule->shift) : '');

        $event->start_date_time = $start->format('Y-m-d H:i:s');
        $event->end_date_time = $end->format('Y-m-d H:i:s');

        $event->last_updated_by = user() ? user()->id : null;

        $event->save();
    }

    protected function deleteCalendarEventForSchedule(Schedule $schedule): void
    {
        if (!class_exists(\App\Models\Event::class)) {
            return;
        }

        $companyId = $schedule->company_id ?? (company() ? company()->id : null);

        Event::where('scheduled_service_id', $schedule->id)
            ->where('event_type', 'inspection')
            ->when($companyId, fn($q) => $q->where('company_id', $companyId))
            ->delete();
    }


    public function download($id)
    {
        $this->schedule = Schedule::findOrFail($id);

        $this->viewPermission = user()->permission('view_inspection');
        $this->company = $this->schedule->company;

        // Download file uploaded
        if ($this->schedule->file != null) {
            return response()->download(storage_path('app/public/schedule-files') . '/' . $this->schedule->file);
        }

        $pdfOption = $this->domPdfObjectForDownload($id);
        $pdf = $pdfOption['pdf'];
        $filename = $pdfOption['fileName'];

        return request()->view ? $pdf->stream($filename . '.pdf') : $pdf->download($filename . '.pdf');
    }

    public function domPdfObjectForDownload($id)
    {
        $this->schedule = Schedule::findOrFail($id);

        $this->company = $this->schedule->company;

        $pdf = app('dompdf.wrapper');
        $pdf->setOption('enable_php', true);
        $pdf->setOption('isHtml5ParserEnabled', true);
        $pdf->setOption('isRemoteEnabled', true);

        $pdf->loadView('inspection_schedules.pdf.' . $this->scheduleSetting->template, $this->data);
        $filename = $this->schedule->schedule_number;

        return [
            'pdf' => $pdf,
            'fileName' => $filename
        ];
    }

    public function domPdfObjectForConsoleDownload($id)
    {
        $this->schedule = Schedule::findOrFail($id);

        $this->company = $this->schedule->company;

        $pdf = app('dompdf.wrapper');
        $pdf->setOption('enable_php', true);
        $pdf->setOptions(['isHtml5ParserEnabled' => true, 'isRemoteEnabled' => true]);

        App::setLocale($this->scheduleSetting->locale);
        Carbon::setLocale($this->scheduleSetting->locale);
        $pdf->loadView('inspection_schedules.pdf.schedule-recurring', $this->data);

        $dom_pdf = $pdf->getDomPDF();
        $canvas = $dom_pdf->getCanvas();
        $canvas->page_text(530, 820, 'Page {PAGE_NUM} of {PAGE_COUNT}', null, 10);

        $filename = $this->schedule->remark;

        return [
            'pdf' => $pdf,
            'fileName' => $filename
        ];
    }



    public function cancelStatus(Request $request)
    {
        $schedule = Schedule::findOrFail($request->scheduleID);
        $schedule->status = 'canceled'; // update status as canceled
        $schedule->save();

        // Sync to WorkSuite calendar
        $this->upsertCalendarEventForSchedule($schedule);

        return Reply::success(__('messages.updateSuccess'));
    }

}
