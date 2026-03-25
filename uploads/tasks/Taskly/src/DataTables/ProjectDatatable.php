<?php

namespace Workdo\Taskly\DataTables;

use Illuminate\Database\Eloquent\Builder as QueryBuilder;
use Illuminate\Http\Request;
use Workdo\Taskly\Entities\Project;
use Yajra\DataTables\EloquentDataTable;
use Yajra\DataTables\Html\Builder as HtmlBuilder;
use Yajra\DataTables\Html\Button;
use Yajra\DataTables\Html\Column;
use Yajra\DataTables\Html\Editor\Editor;
use Yajra\DataTables\Html\Editor\Fields;
use Yajra\DataTables\Services\DataTable;

class ProjectDatatable extends DataTable
{
    /**
     * Build the DataTable class.
     *
     * @param QueryBuilder $query Results from query() method.
     */
    public function dataTable(QueryBuilder $query): EloquentDataTable
    {
        $rowcolumn = ['name','status','start_date','end_date','assign_user'];
        $dataTable = (new EloquentDataTable($query))
        ->editColumn('start_date',function(Project $project){
            return company_date_formate($project->start_date);
        })
        ->editColumn('end_date',function(Project $project){
            return company_date_formate($project->end_date);
        })
        ->editColumn('assign_user',function(Project $project){
            $html ='';
            foreach ($project->users as $user)
            {
                if (check_file($user->avatar) == false) {
                    $path = asset('uploads/user-avatar/avatar.png');
                } else {
                    $path = get_file($user->avatar);
                }
                $html .= '<img  src="' . $path . '" data-bs-toggle="tooltip"  title=" ' .$user->name . ' " data-bs-placement="top"  class="rounded-circle" width="25" height="25">';
            }
            return $html;
        });
        if (\Laratrust::hasPermission('task manage') || \Laratrust::hasPermission('project create') || \Laratrust::hasPermission('project show') || \Laratrust::hasPermission('project edit') ||\Laratrust::hasPermission('project delete')) {
            $dataTable->addColumn('action', function (Project $project) {
                return view('taskly::projects.action', compact('project'));
            });
            $rowcolumn[] = 'action';
        }
        return $dataTable->rawColumns($rowcolumn);
    }

    /**
     * Get the query source of dataTable.
     */
    public function query(Project $model, Request $request)
    {
        $user = \Auth::user();
        $workspace = getActiveWorkSpace();

        if ($user->hasRole('client')) {
            $projects = $model->select('projects.*')
                ->join('client_projects', 'projects.id', '=', 'client_projects.project_id')
                ->where('client_projects.client_id', '=', $user->id)
                ->where('projects.workspace', '=', $workspace)
                ->projectonly()
                ->groupBy('projects.id');
        } else {
            $projects = $model->select('projects.*')
                ->join('user_projects', 'projects.id', '=', 'user_projects.project_id')
                ->where('user_projects.user_id', '=', $user->id)
                ->where('projects.workspace', '=', $workspace)
                ->projectonly()
                ->groupBy('projects.id');
        }

        if (!empty($request->start_date)) {
            $projects = $projects->where('start_date', $request->start_date);
        }
        if (!empty($request->end_date)) {
            $projects = $projects->where('end_date', $request->end_date);
        }
        return $projects;
    }

    /**
     * Optional method if you want to use the html builder.
     */
    public function html(): HtmlBuilder
    {
        $dataTable = $this->builder()
            ->setTableId('project-table')
            ->columns($this->getColumns())
            ->ajax([
                'data' => 'function(d) {
                    var start_date = $("input[name=start_date]").val();
                    d.start_date = start_date

                    var end_date = $("input[name=end_date]").val();
                    d.end_date = end_date
                }',
            ])
            ->orderBy(0)
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
                $("body").on("click", "#applyfilter", function() {
                    if (!$("input[name=start_date]").val() && !$("input[name=end_date]").val()) {
                        toastrs("Error!", "Please select Atleast One Filter ", "error");
                        return;
                    }

                    $("#project-table").DataTable().draw();
                });

                $("body").on("click", "#clearfilter", function() {
                    $("input[name=start_date]").val("")
                    $("input[name=end_date]").val("")
                    $("#project-table").DataTable().draw();
                });

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
            Column::make('id')->searchable(false)->visible(false)->exportable(false)->printable(false),
            Column::make('name')->title(__('Name')),
            Column::make('status')->title(__('Stage')),
            Column::make('start_date')->title(__('Start Date')),
            Column::make('end_date')->title(__('End Date')),
            Column::make('assign_user')->title(__('Assign User'))->searchable(false)->printable(false)
        ];
        if (\Laratrust::hasPermission('task manage') || \Laratrust::hasPermission('project create') || \Laratrust::hasPermission('project show') || \Laratrust::hasPermission('project edit') ||\Laratrust::hasPermission('project delete')) {
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
        return 'Project_' . date('YmdHis');
    }
}
