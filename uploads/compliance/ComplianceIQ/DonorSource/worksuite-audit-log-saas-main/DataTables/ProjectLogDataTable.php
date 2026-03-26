<?php

namespace Modules\AuditLog\DataTables;

use Carbon\Carbon;
use App\DataTables\BaseDataTable;
use App\ProjectActivity;
use Yajra\DataTables\Html\Button;

class ProjectLogDataTable extends BaseDataTable
{
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
            ->addColumn('projects.project_name', function ($row) {
                return '<a target="_blank" href="' . route("admin.projects.show", $row->project_id) . '">' . $row->project_name . '</a>';
            })
            ->addColumn('activity', function ($row) {
                return $row->activity;
            })
            ->editColumn('created_at', function ($row) {
                return $row->created_at->format('d M Y - h:i a');
            })
            ->rawColumns(['projects.project_name']);
    }

    /**
     * Get query source of dataTable.
     *
     * @return \Illuminate\Database\Eloquent\Builder
     */
    public function query()
    {
        $model = ProjectActivity::join('projects', 'projects.id', 'project_activity.project_id');

        $date = Carbon::create((request()->year ?? date('Y')) . '-' . (request()->month ?? date('m')) . '-01');
        $startDate = $date->copy()->startOfMonth();
        $endDate = $date->copy()->endOfMonth();
        $model->whereBetween('project_activity.created_at', [$startDate, $endDate]);

        return $model;
    }

    /**
     * Optional method if you want to use html builder.
     *
     * @return \Yajra\DataTables\Html\Builder
     */
    public function html()
    {
        return $this->builder()
            ->setTableId('project_log')
            ->columns($this->getColumns())
            ->minifiedAjax()
            ->dom("<'row'<'col-md-6'l><'col-md-6'Bf>><'row'<'col-sm-12'tr>><'row'<'col-sm-5'i><'col-sm-7'p>>")
            ->orderBy(0)
            ->destroy(true)
            ->responsive(true)
            ->serverSide(true)
            ->stateSave(true)
            ->processing(true)
            ->language(__("app.datatable"))
            ->buttons(
                Button::make()
            )
            ->parameters([
                'initComplete' => 'function () {
                   window.LaravelDataTables["project_log"].buttons().container()
                    .appendTo( ".bg-title .text-right")
                }',
                'fnDrawCallback' => 'function( oSettings ) {
                    $("body").tooltip({
                        selector: \'[data-toggle="tooltip"]\'
                    })
                }',
            ]);
    }

    /**
     * Get columns.
     *
     * @return array
     */
    protected function getColumns()
    {
        return [
            __('app.id') => ['data' => 'project_id', 'name' => 'project_id'],
            __('project_name') => ['data' => 'projects.project_name', 'name' => 'projects.project_name'],
            __('activity') => ['data' => 'activity', 'name' => 'activity'],
            __('app.date') => ['data' => 'created_at', 'name' => 'created_at'],
        ];
    }

    /**
     * Get filename for export.
     *
     * @return string
     */
    protected function filename()
    {
        return 'task_log_' . date('YmdHis');
    }

    public function pdf()
    {
        set_time_limit(0);
        if ('snappy' == config('datatables-buttons.pdf_generator', 'snappy')) {
            return $this->snappyPdf();
        }

        $pdf = app('dompdf.wrapper');
        $pdf->loadView('datatables::print', ['data' => $this->getDataForPrint()]);

        return $pdf->download($this->getFilename() . '.pdf');
    }
}
