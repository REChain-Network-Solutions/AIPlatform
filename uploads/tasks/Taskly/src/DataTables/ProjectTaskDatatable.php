<?php

namespace Workdo\Taskly\DataTables;

use Illuminate\Database\Eloquent\Builder as QueryBuilder;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Workdo\Taskly\Entities\Task;
use Yajra\DataTables\EloquentDataTable;
use Yajra\DataTables\Html\Builder as HtmlBuilder;
use Yajra\DataTables\Html\Button;
use Yajra\DataTables\Html\Column;
use Yajra\DataTables\Html\Editor\Editor;
use Yajra\DataTables\Html\Editor\Fields;
use Yajra\DataTables\Services\DataTable;

class ProjectTaskDatatable extends DataTable
{
    /**
     * Build the DataTable class.
     *
     * @param QueryBuilder $query Results from query() method.
     */
    public function dataTable(QueryBuilder $query): EloquentDataTable
    {
        $rowcolumn = ['title','milestone_id','priority','status','assign_to'];
        $dataTable = (new EloquentDataTable($query))
        ->editColumn('milestone_id',function(Task $task){
            return !empty($task->milestone_name) ?$task->milestone_name :'-';
        })
        ->editColumn('status',function(Task $task){
            return $task->stage_name;
        })
        ->editColumn('assign_to',function(Task $task){
            $html ='';
            foreach ($task->users() as $user)
            {
                if (check_file($user->avatar) == false) {
                    $path = asset('uploads/user-avatar/avatar.png');
                } else {
                    $path = get_file($user->avatar);
                }
                $html .= '<img  src="' . $path . '" data-bs-toggle="tooltip"  title="' . $user->name . '" data-bs-placement="top"  class="rounded-circle" width="25" height="25">';
            }
            return $html;
        });
        if (\Laratrust::hasPermission('task show') || \Laratrust::hasPermission('task edit') || \Laratrust::hasPermission('task delete')) {
            $dataTable->addColumn('action', function (Task $task) {
                return view('taskly::projects.task_action', compact('task'));
            });
            $rowcolumn[] = 'action';
        }
        return $dataTable->rawColumns($rowcolumn);

    }

    /**
     * Get the query source of dataTable.
     */
    public function query(Task $model)
    {
        $projectid = \Request::segment(2);
        $task = $model->select('tasks.*' ,'stages.name as stage_name' ,'milestones.title as milestone_name')
        ->join('stages','stages.id', '=', 'tasks.status')
        ->leftJoin('milestones','milestones.id', '=', 'tasks.milestone_id')
        ->where('tasks.project_id', $projectid)
        ->where('tasks.workspace', getActiveWorkSpace())
        ->groupBy('tasks.id');
        $objUser = Auth::user();
        if (!Auth::user()->hasRole('client') && !Auth::user()->hasRole('company')) {
            if (isset($objUser) && $objUser) {
                $task->whereRaw("find_in_set('" . $objUser->id . "',assign_to)");
            }
        }
       return $task;
    }

    /**
     * Optional method if you want to use the html builder.
     */
    public function html(): HtmlBuilder
    {
        $dataTable = $this->builder()
            ->setTableId('projects-task-table')
            ->columns($this->getColumns())
            ->minifiedAjax()
            ->orderBy(1)
            ->language([
                "paginate" => [
                    "next" => '<i class="ti ti-chevron-right"></i>',
                    "previous" => '<i class="ti ti-chevron-left"></i>'
                ],
                'lengthMenu' => "_MENU_" . __('Entries Per Page'),
                "searchPlaceholder" => __('Search...'),
                "search" => "",
                "info" => __('Showing _START_ to _END_ of _TOTAL_ entries')
            ])
            ->initComplete('function() {
                var table = this;
                var searchInput = $(\'#\'+table.api().table().container().id+\' label input[type="search"]\');
                searchInput.removeClass(\'form-control form-control-sm\');
                searchInput.addClass(\'dataTable-input\');
                var select = $(table.api().table().container()).find(".dataTables_length select").removeClass(\'custom-select custom-select-sm form-control form-control-sm\').addClass(\'dataTable-selector\');
            }');

        $exportButtonConfig = [
            'extend' => 'collection',
            'className' => 'btn btn-light-secondary dropdown-toggle',
            'text' => '<i class="ti ti-download me-2" data-bs-toggle="tooltip" data-bs-placement="top" data-bs-original-title="Export"></i>',
            'buttons' => [
                [
                    'extend' => 'print',
                    'text' => '<i class="fas fa-print me-2"></i> ' . __('Print'),
                    'className' => 'btn btn-light text-primary dropdown-item',
                    'exportOptions' => ['columns' => [0, 1, 3]],
                ],
                [
                    'extend' => 'csv',
                    'text' => '<i class="fas fa-file-csv me-2"></i> ' . __('CSV'),
                    'className' => 'btn btn-light text-primary dropdown-item',
                    'exportOptions' => ['columns' => [0, 1, 3]],
                ],
                [
                    'extend' => 'excel',
                    'text' => '<i class="fas fa-file-excel me-2"></i> ' . __('Excel'),
                    'className' => 'btn btn-light text-primary dropdown-item',
                    'exportOptions' => ['columns' => [0, 1, 3]],
                ],
            ],
        ];

        $buttonsConfig = array_merge([
            $exportButtonConfig,
            [
                'extend' => 'reset',
                'className' => 'btn btn-light-danger',
            ],
            [
                'extend' => 'reload',
                'className' => 'btn btn-light-warning',
            ],
        ]);

        $dataTable->parameters([
            "dom" =>  "
        <'dataTable-top'<'dataTable-dropdown page-dropdown'l><'dataTable-botton table-btn dataTable-search tb-search  d-flex justify-content-end gap-2'Bf>>
        <'dataTable-container'<'col-sm-12'tr>>
        <'dataTable-bottom row'<'col-5'i><'col-7'p>>",
            'buttons' => $buttonsConfig,
            "drawCallback" => 'function( settings ) {
                var tooltipTriggerList = [].slice.call(
                    document.querySelectorAll("[data-bs-toggle=tooltip]")
                  );
                  var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                    return new bootstrap.Tooltip(tooltipTriggerEl);
                  });
                  var popoverTriggerList = [].slice.call(
                    document.querySelectorAll("[data-bs-toggle=popover]")
                  );
                  var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
                    return new bootstrap.Popover(popoverTriggerEl);
                  });
                  var toastElList = [].slice.call(document.querySelectorAll(".toast"));
                  var toastList = toastElList.map(function (toastEl) {
                    return new bootstrap.Toast(toastEl);
                  });
            }'
        ]);

        $dataTable->language([
            'buttons' => [
                'create' => __('Create'),
                'export' => __('Export'),
                'print' => __('Print'),
                'reset' => __('Reset'),
                'reload' => __('Reload'),
                'excel' => __('Excel'),
                'csv' => __('CSV'),
            ]
        ]);

        return $dataTable;
    }

    /**
     * Get the dataTable columns definition.
     */
    public function getColumns(): array
    {
        $column = [
            Column::make('title')->title(__('Title')),
            Column::make('milestone_id')->title(__('Milestone'))->name('milestones.title'),
            Column::make('priority')->title(__('Priority')),
            Column::make('status')->title(__('Stage'))->name('stages.name'),
            Column::make('assign_to')->title(__('Assign User'))->exportable(false)->searchable(false)->printable(false)
        ];
        if (\Laratrust::hasPermission('task show') || \Laratrust::hasPermission('task edit') || \Laratrust::hasPermission('task delete')) {
            $action = [
                Column::computed('action')
                ->exportable(false)
                ->printable(false)
                ->width(60)
                
            ];
            $column = array_merge($column,$action);
        }
        return $column;
    }

    /**
     * Get the filename for export.
     */
    protected function filename(): string
    {
        return 'Task_' . date('YmdHis');
    }
}
