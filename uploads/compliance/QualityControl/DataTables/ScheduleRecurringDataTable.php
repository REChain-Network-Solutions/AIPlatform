<?php

namespace Modules\QualityControl\DataTables;

use Carbon\Carbon;
use App\Models\User;
use App\Scopes\CompanyScope;
use App\DataTables\BaseDataTable;
use Yajra\DataTables\Html\Button;
use Yajra\DataTables\Html\Column;
use Illuminate\Support\Facades\DB;
use Modules\QualityControl\Entities\RecurringSchedule;

class ScheduleRecurringDataTable extends BaseDataTable
{
    private $viewSchedulePermission;
    private $deleteSchedulePermission;
    private $editSchedulePermission;

    public function __construct()
    {
        parent::__construct();
        $this->viewSchedulePermission = user()->permission('view_quality_control');
        $this->deleteSchedulePermission = user()->permission('delete_quality_control');
        $this->editSchedulePermission = user()->permission('edit_quality_control');

    }

    /**
     * Build DataTable class.
     *
     * @param mixed $query Results from query() method.
     * @return \Yajra\DataTables\DataTableAbstract
     */
    public function dataTable($query)
    {
        return datatables()
            ->eloquent($query)
            ->addIndexColumn()
            ->addColumn('action', function ($row) {
                $action = '<div class="task_view">

                <div class="dropdown">
                    <a class="task_view_more d-flex align-items-center justify-content-center dropdown-toggle" type="link"
                        id="dropdownMenuLink-' . $row->id . '" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                        <i class="icon-options-vertical icons"></i>
                    </a>
                    <div class="dropdown-menu dropdown-menu-right" aria-labelledby="dropdownMenuLink-' . $row->id . '" tabindex="0">';

                $action .= '<a href="' . route('recurring-inspection_schedules.show', [$row->id]) . '" class="dropdown-item"><i class="fa fa-eye mr-2"></i>' . __('app.view') . '</a>';

                if ($this->editSchedulePermission == 'all' ) {
                    $action .= '<a class="dropdown-item" href="' . route('recurring-inspection_schedules.edit', $row->id) . '" >
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
                return $row->subject;
            })
            ->addColumn('status', function ($row) {

                    $selectActive = $row->status == 'active' ? 'selected' : '';
                    $selectInactive = $row->status != 'active' ? 'selected' : '';

                    $role = '<select class="form-control select-picker change-schedule-status" data-schedule-id="' . $row->id . '">';

                    $role .= '<option data-content="<i class=\'fa fa-circle mr-2 text-light-green\'></i> ' . __('app.active') . '" value="active" ' . $selectActive . '> ' . __('app.active') . ' </option>';
                    $role .= '<option data-content="<i class=\'fa fa-circle mr-2 text-red\'></i> ' . __('app.inactive') . '" value="inactive" ' . $selectInactive . '> ' . __('app.inactive') . ' </option>';

                    $role .= '</select>';

                    return $role;
            })
            ->editColumn(
                'issue_date',
                function ($row) {
                    return $row->issue_date->timezone($this->company->timezone)->translatedFormat($this->company->date_format);
            })
            ->editColumn(
                'next_schedule_date',
                function ($row) {
                    $rotation = '<span class="px-1"><label class="badge badge-' . RecurringSchedule::ROTATION_COLOR[$row->rotation] . '">' . $row->rotation . '</label></span';

                    if (is_null($row->next_schedule_date)) {
                        return $rotation;
                    }

                    $date = $row->next_schedule_date->timezone($this->company->timezone)->translatedFormat($this->company->date_format);

                    return $date . $rotation;

                }
            )
            ->rawColumns([ 'action', 'status','next_schedule_date']);
    }

    /**
     * @param RecurringSchedule $model
     * @return $this|RecurringSchedule
     */
    public function query(RecurringSchedule $model)
    {
        $request = $this->request();

        $model = $model->select('*');

        if ($request->startDate !== null && $request->startDate != 'null' && $request->startDate != '') {
            $startDate = Carbon::createFromFormat($this->company->date_format, $request->startDate)->toDateString();
            $model = $model->where(DB::raw('DATE(inspection_schedule_recurring.`issue_date`)'), '>=', $startDate);
        }

        if ($request->endDate !== null && $request->endDate != 'null' && $request->endDate != '') {
            $endDate = Carbon::createFromFormat($this->company->date_format, $request->endDate)->toDateString();
            $model = $model->where(DB::raw('DATE(inspection_schedule_recurring.`issue_date`)'), '<=', $endDate);
        }

        if ($request->status != 'all' && !is_null($request->status)) {
            $model = $model->where('inspection_schedule_recurring.status', '=', $request->status);
        }


        if ($request->searchText != '') {
            $model = $model->where(function ($query) {
                $query->where('inspection_schedule_recurring.subject', 'like', '%' . request('searchText') . '%')
                    ->orWhere('inspection_schedule_recurring.lokasi', 'like', '%' . request('searchText') . '%');
            });
        }


        // $model = $model->whereHas('subject', function ($q) {
        //     $q->whereNull('deleted_at');
        // }, '>=', 0);

        return $model;
    }

    /**
     * Optional method if you want to use html builder.
     *
     * @return \Yajra\DataTables\Html\Builder
     */
    public function html()
    {
        return $this->setBuilder('inspection_schedules-recurring-table', 0)
            ->parameters([
                'initComplete' => 'function () {
                   window.LaravelDataTables["inspection_schedules-recurring-table"].buttons().container()
                    .appendTo("#table-actions")
                }',
                'fnDrawCallback' => 'function( oSettings ) {
                    $(".change-schedule-status").selectpicker();
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
            '#' => ['data' => 'id', 'orderable' => false, 'searchable' => false],
            __('Schedule Job') => ['data' => 'subject', 'name' => 'subject', 'exportable' => false, 'title' => __('Schedule Job')],
            __('modules.inspection_schedules.startDate') => ['data' => 'issue_date', 'name' => 'issue_date', 'title' => __('app.startDate')],
            __('Next Schedule') => ['data' => 'next_schedule_date', 'name' => 'next_schedule_date', 'title' => __('Next Schedule')],
            __('app.status') => ['data' => 'status', 'name' => 'status', 'exportable' => false, 'title' => __('app.status')],
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
