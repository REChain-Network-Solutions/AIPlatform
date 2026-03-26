<?php

namespace Modules\QualityControl\DataTables;

use Carbon\Carbon;


use App\DataTables\BaseDataTable;
use Yajra\DataTables\Html\Button;
use Yajra\DataTables\Html\Column;
use Illuminate\Support\Facades\DB;
use Yajra\DataTables\EloquentDataTable;
use Modules\QualityControl\Entities\Schedule;

class RecurringSchedulesDataTable extends BaseDataTable
{

    protected $firstSchedule;

    private $viewSchedulePermission;

    public function dataTable($query)
    {
        $firstSchedule = $this->firstSchedule;
        $scheduleSettings = $this->scheduleSettings;

        return (new EloquentDataTable($query))
            ->addIndexColumn()
            ->addColumn('action', function ($row) use ($firstSchedule) {

                $action = '<div class="task_view">

                <div class="dropdown">
                    <a class="task_view_more d-flex align-items-center justify-content-center dropdown-toggle" type="link"
                        id="dropdownMenuLink-' . $row->id . '" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                        <i class="icon-options-vertical icons"></i>
                    </a>
                    <div class="dropdown-menu dropdown-menu-right" aria-labelledby="dropdownMenuLink-' . $row->id . '" tabindex="0">';

                $action .= '<a href="' . route('schedules.show', [$row->id]) . '" class="dropdown-item"><i class="fa fa-eye mr-2"></i>' . __('app.view') . '</a>';





                $action .= '</div>
                </div>
            </div>';

                return $action;
            })
            ->editColumn('subject', function ($row) {
                if ($row->subject != null) {
                    return $row->subject;
                }

                return '--';
            })


            ->editColumn(
                'issue_date',
                function ($row) {
                    return $row->issue_date->timezone($this->company->timezone)->translatedFormat($this->company->date_format);
                }
            )
            ->editColumn('status', function ($row) {
                if ($row->status != null) {
                    return $row->status;
                }

                return '--';
            })
            ->editColumn('priority', function ($row) {
                if ($row->priority != null) {
                    return $row->priority;
                }

                return '--';
            })
            ->rawColumns([ 'action'])
           ;
    }

    /**
     * @param Schedule $model
     * @return \Illuminate\Database\Eloquent\Builder
     */
    public function query(Schedule $model)
    {
        $request = $this->request();

        $this->firstSchedule = Schedule::orderBy('id', 'desc')->first();


        $model =  $model->select('id',  'subject', 'issue_date', 'status', 'priority');

        if ($request->startDate !== null && $request->startDate != 'null' && $request->startDate != '') {
            $startDate = Carbon::createFromFormat($this->company->date_format, $request->startDate)->toDateString();
            $model = $model->where(DB::raw('DATE(inspection_schedules.`issue_date`)'), '>=', $startDate);
        }

        if ($request->endDate !== null && $request->endDate != 'null' && $request->endDate != '') {
            $endDate = Carbon::createFromFormat($this->company->date_format, $request->endDate)->toDateString();
            $model = $model->where(DB::raw('DATE(inspection_schedules.`issue_date`)'), '<=', $endDate);
        }

        if ($request->status != 'all' && !is_null($request->status)) {
            $model = $model->where('inspection_schedules.status', '=', $request->status);
        }





        if ($request->searchText != '') {
            $model = $model->where(function ($query) {
                $query->where('inspection_schedules.subject', 'like', '%' . request('searchText') . '%')
                    ->orWhere('inspection_schedules.pic', 'like', '%' . request('searchText') . '%')
                    ->orWhere('inspection_schedules.remark', 'like', '%' . request('searchText') . '%');
            });
        }

        $model = $model->where('schedule_recurring_id', '=', $request->recurringID);

        $model = $model->whereNull('deleted_at');

        return $model;
    }

    /**
     * Optional method if you want to use html builder.
     *
     * @return \Yajra\DataTables\Html\Builder
     */
    public function html()
    {
        return $this->setBuilder('recurring-inspection_schedules-table', 0)
            ->parameters([
                'initComplete' => 'function () {
                   window.LaravelDataTables["recurring-inspection_schedules-table"].buttons().container()
                    .appendTo("#table-actions")
                }',
                'fnDrawCallback' => 'function( oSettings ) {
                    $("body").tooltip({
                        selector: \'[data-toggle="tooltip"]\'
                    })
                }',
            ])
            ->buttons(Button::make(['extend' => 'excel', 'text' => '<i class="fa fa-file-export"></i> ' . trans('app.exportExcel')]));
    }

    /**
     * Get columns.
     *
     * @return array
     */
    protected function getColumns()
    {
        $modules = $this->user->modules;

        $dsData = [
            __('app.id') => ['data' => 'id', 'name' => 'id', 'visible' => false, 'title' => __('app.id')],
            __('Subject') . '#' => ['data' => 'subject', 'name' => 'subject', 'title' => __('Subject')],
            __('Issued Date') . '#' => ['data' => 'issue_date', 'name' => 'issue_date', 'title' => __('Issued Date')],
            __('Status') => ['data' => 'status', 'name' => 'status', 'title' => __('Status')],
            __('Priority') => ['data' => 'priority', 'name' => 'priority', 'title' => __('Priority')],
            Column::computed('action', __('app.action'))
                ->exportable(false)
                ->printable(false)
                ->orderable(false)
                ->searchable(false)
                ->width(150)
                ->addClass('text-right pr-20')
        ];


        return $dsData;
    }



}
