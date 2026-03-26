<?php

namespace Modules\QualityControl\DataTables;

use Carbon\Carbon;


use App\DataTables\BaseDataTable;
use Yajra\DataTables\Html\Button;
use Yajra\DataTables\Html\Column;
use Illuminate\Support\Facades\DB;
use Yajra\DataTables\EloquentDataTable;
use Modules\QualityControl\Entities\Schedule;

class SchedulesDataTable extends BaseDataTable
{

    protected $firstSchedule;

    private $viewSchedulePermission;
    private $deleteSchedulePermission;
    private $editSchedulePermission;

    public function dataTable($query)
    {
        $firstSchedule = $this->firstSchedule;
        $scheduleSettings = $this->scheduleSettings;

        $this->viewSchedulePermission = user()->permission('view_quality_control');
        $this->deleteSchedulePermission = user()->permission('delete_quality_control');
        $this->editSchedulePermission = user()->permission('edit_quality_control');

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

                if ($this->editSchedulePermission == 'all' ) {
                    $action .= '<a class="dropdown-item" href="' . route('schedules.edit', $row->id) . '" >
                                <i class="fa fa-edit mr-2"></i>
                                ' . trans('app.edit') . '
                            </a>';
                }

                if ($this->deleteSchedulePermission == 'all' ) {
                    $action .= '<a class="dropdown-item delete-schedule" href="javascript:;" data-toggle="tooltip"  data-schedule-id="' . $row->id . '">
                                    <i class="fa fa-trash mr-2"></i>
                                    ' . trans('app.delete') . '
                                </a>';
                }



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
            ->editColumn(
                'worker_id',
                function ($row) {
                    if (is_null($row->worker)) {
                        return '';
                    }

                    if ($row->worker->hasRole('employee')) {
                        return view('components.employee', [
                            'user' => $row->worker
                        ]);
                    }
            })
            ->editColumn('status', function ($row) {
                if ($row->status != null) {
                    return $row->status;
                }

                return '--';
            })

            ->rawColumns([ 'action', 'worker_id'])
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


        $model =  $model->select('id',  'subject', 'issue_date', 'worker_id', 'status');

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


        return $model;
    }

    /**
     * Optional method if you want to use html builder.
     *
     * @return \Yajra\DataTables\Html\Builder
     */
    public function html()
    {
        return $this->setBuilder('inspection_schedules-table', 0)
            ->parameters([
                'initComplete' => 'function () {
                   window.LaravelDataTables["inspection_schedules-table"].buttons().container()
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
            __('Worker') => ['data' => 'worker_id', 'name' => 'worker_id', 'title' => __('Worker'), 'width' => '20%'],
            __('Status') => ['data' => 'status', 'name' => 'status', 'title' => __('Status')],
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
