<?php

namespace Modules\AuditLog\DataTables;

use Carbon\Carbon;
use App\DataTables\BaseDataTable;
use App\LogActivity;
use Yajra\DataTables\Html\Button;
use Illuminate\Support\Str;

class LogActivityDataTable extends BaseDataTable
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
            ->addColumn('user.name', function ($row) {
                return $row->user ? '<a target="_blank" href="' . route("admin.employees.show", $row->causer_id) . '">' . $row->user->name . '</a>' : '-';
            })
            ->editColumn('subject_type', function ($row) {
                return  Str::afterLast($row->subject_type, '\\') . ' Model';
            })
            ->editColumn('description', function ($row) {
                return  $row->description;
            })
            ->editColumn('properties', function ($row) {
                if ($row->properties && $row->log_name != 'created' && $row->log_name != 'created') {
                    $global = global_settings();
                    return view('auditlog::properties')->with('properties', json_decode($row->properties, true))->with('id', $row->id)->with('global', $global);
                } else
                    return '--';
            })
            ->editColumn('created_at', function ($row) {
                return $row->created_at->format('d M Y - h:i a');
            })
            ->rawColumns(['user.name', 'properties']);
    }

    /**
     * Get query source of dataTable.
     *
     * @return \Illuminate\Database\Eloquent\Builder
     */
    public function query()
    {
        $logActivity = LogActivity::leftJoin('users', 'users.id', '=', 'log_activities.causer_id')
            ->select('log_activities.*', 'users.name as name');

        if (request()->daterange)
       {
        $dates = explode(' - ', request()->daterange);
        $startCreate = now()->subMonth()->format($this->global->date_format);
        $endCreate = now()->format($this->global->date_format);
        $startDate = Carbon::createFromFormat($this->global->date_format, $dates[0] ?? $startCreate);
        $endDate = Carbon::createFromFormat($this->global->date_format, $dates[1] ?? $endCreate);

        $logActivity = $logActivity->whereBetween('log_activities.created_at', [$startDate->toDateString() . ' 00:00:00', $endDate->toDateString() . ' 23:59:59']);
       }


        if (request()->model_name)
            $logActivity = $logActivity->where('log_activities.subject_type', request()->model_name);

        return $logActivity;
    }

    /**
     * Optional method if you want to use html builder.
     *
     * @return \Yajra\DataTables\Html\Builder
     */
    public function html()
    {
        return $this->builder()
            ->setTableId('log_activity')
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
                   window.LaravelDataTables["log_activity"].buttons().container()
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
            __('auditlog::app._log_activity.id')           => ['data' => 'id', 'name' => 'log_activities.id'],
            __('auditlog::app._log_activity.model')        => ['data' => 'subject_type', 'name' => 'log_activities.subject_type'],
            __('auditlog::app._log_activity.user')         => ['data' => 'name', 'name' => 'users.name'],
            __('auditlog::app._log_activity.activity')     => ['data' => 'description', 'name' => 'log_activities.description'],
            __('auditlog::app._log_activity.properties')   => ['data' => 'properties', 'name' => 'log_activities.properties'],
            __('auditlog::app._log_activity.ip')           => ['data' => 'ip', 'name' => 'log_activities.ip'],
            __('auditlog::app._log_activity.date')         => ['data' => 'created_at', 'name' => 'log_activities.created_at'],
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
