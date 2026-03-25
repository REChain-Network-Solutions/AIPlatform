<?php

namespace Workdo\Pos\DataTables;

use Illuminate\Database\Eloquent\Builder as QueryBuilder;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Yajra\DataTables\EloquentDataTable;
use Yajra\DataTables\Html\Builder as HtmlBuilder;
use Yajra\DataTables\Html\Button;
use Yajra\DataTables\Html\Column;
use Yajra\DataTables\Html\Editor\Editor;
use Yajra\DataTables\Html\Editor\Fields;
use Yajra\DataTables\Services\DataTable;
use Workdo\Pos\Entities\Pos;

class PosOrderDataTable extends DataTable
{
    /**
     * Build the DataTable class.
     *
     * @param QueryBuilder $query Results from query() method.
     */
    public function dataTable(QueryBuilder $query): EloquentDataTable
    {
        $rowColumn = ['pos_date','customer_id','warehouse_id','amount','id'];
        $dataTable =  (new EloquentDataTable($query))
            ->addIndexColumn()
            ->editColumn('id', function (Pos $model) {
                if (\Laratrust::hasPermission(['pos show'])) {
                    $html = "<a href='" . route('pos.show', \Crypt::encrypt($model->id)) . "' class='btn btn-outline-primary'>" . Pos::posNumberFormat($model->pos_id) . "</a>";
                } else {
                    $html =  "<a href='!#' class='btn btn-outline-primary'>" .Pos::posNumberFormat($model->pos_id) . "</a>";
                }
                return $html;
            })
            ->filterColumn('id', function ($query, $keyword){
                $formattedValue = str_replace(!empty(company_setting('pos_prefix')) ? company_setting('pos_prefix') : '#POS', '', $keyword);
                $query->where('id', $formattedValue);
                $query->orWhereHas('posPayment', function ($q) use ($keyword) {
                    $q->where('amount', 'like', "%$keyword%");
                });
            })
            ->editColumn('pos_date', function (Pos $model) {
                return company_date_formate($model->pos_date);

            })
            ->editColumn('customer_id', function (Pos $model) {
                if($model->customer_id == 0){
                    $html = 'Walk-in Customer';
                }elseif(empty($posPayment->customer)){
                    $html = isset($model->user)?$model->user->name:'';
                }else{
                    $html = isset($model->customer)?$model->customer->name:'';
                }
                return $html;
            })
            ->filterColumn('customer_id', function ($query, $keyword) {
                $query->whereHas('customer', function ($q) use ($keyword) {
                    $q->where('name', 'like', "%$keyword%");
                });
            })
            ->editColumn('warehouse_id', function (Pos $model) {
                return isset($model->warehouse)?$model->warehouse->name:'-';
            })
            ->filterColumn('warehouse_id', function ($query, $keyword) {
                $query->whereHas('warehouse', function ($q) use ($keyword) {
                    $q->where('name', 'like', "%$keyword%");
                });
            })
            ->editColumn('amount', function (Pos $model) {
                return isset($model->posPayment)?currency_format_with_sym($model->posPayment->amount):currency_format_with_sym(0);
            });
            if (\Laratrust::hasPermission('pos show')) {
                $dataTable->addColumn('action', function (Pos $pos) {
                    return view('pos::pos.action', compact('pos'));
                });
                $rowColumn[] = 'action';
            }
            return $dataTable->rawColumns($rowColumn);
    }

    /**
     * Get the query source of dataTable.
     */
    public function query(Pos $model): QueryBuilder
    {
        if(Auth::user()->type == 'client'){
            $model = $model->where('customer_id', Auth::user()->id);
        }
        return $model->where('created_by', creatorId())->where('workspace',getActiveWorkSpace())->with('customer','warehouse','posPayment');
    }

    /**
     * Optional method if you want to use the html builder.
     */
    public function html(): HtmlBuilder
    {
        $dataTable = $this->builder()
            ->setTableId('pos-table')
            ->columns($this->getColumns())
            ->minifiedAjax()
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
            Column::make('id')->title(__('POS ID')),
            Column::make('pos_date')->title(__('Date')),
            Column::make('customer_id')->title(__('Customer')),
            Column::make('warehouse_id')->title(__('Warehouse')),
            Column::make('amount')->title(__('Amount'))->name('id')->orderable(false)
                ->exportable(false)
                ->printable(false)
        ];

        if (\Laratrust::hasPermission('pos show')) {
            $action = [
                Column::computed('action')
                    ->title(__('Action'))
                    ->exportable(false)
                    ->printable(false)
                    ->width(60)
                    
            ];

            $column = array_merge($column, $action);
        }

        return $column;
    }

    /**
     * Get the filename for export.
     */
    protected function filename(): string
    {
        return 'PosOrder_' . date('YmdHis');
    }
}
