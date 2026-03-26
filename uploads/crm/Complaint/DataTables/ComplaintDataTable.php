<?php

namespace Modules\Complaint\DataTables;

use Carbon\Carbon;
use Modules\Complaint\Entities\Complaint;
use App\Models\CustomField;
use App\Models\CustomFieldGroup;
use App\DataTables\BaseDataTable;
use Illuminate\Database\Eloquent\Model;
use Yajra\DataTables\Html\Button;
use Yajra\DataTables\Html\Column;
use Illuminate\Support\Facades\DB;
use Yajra\DataTables\EloquentDataTable;

class ComplaintDataTable extends BaseDataTable
{
    private $deleteComplaintPermission;
    private $viewComplaintPermission;
    private $editComplaintPermission;

    public function __construct()
    {
        parent::__construct();
        $this->deleteComplaintPermission = user()->permission('delete_complaint');
        $this->viewComplaintPermission = user()->permission('view_complaint');
        $this->editComplaintPermission = user()->permission('edit_complaint');
    }

    /**
     * Build DataTable class.
     *
     * @param mixed $query Results from query() method.
     * @return \Yajra\DataTables\DataTableAbstract
     */
    public function dataTable($query)
    {
        $datatables = datatables()->eloquent($query);
        $datatables->addIndexColumn();
        $datatables->addColumn('action', function ($row) {
            $action = '<div class="task_view">';

            $action .=
                '<div class="dropdown">
                    <a class="task_view_more d-flex align-items-center justify-content-center dropdown-toggle" type="link"
                        id="dropdownMenuLink-' .
                $row->complaint_number .
                '" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                        <i class="icon-options-vertical icons"></i>
                    </a>
                    <div class="dropdown-menu dropdown-menu-right" aria-labelledby="dropdownMenuLink-' .
                $row->complaint_number .
                '" tabindex="0">';

            if ($this->editComplaintPermission == 'all' || $this->editComplaintPermission == 'none' || $this->editComplaintPermission == 'owned') {
                $action .= '<a href="' . route('complaint.show', [$row->complaint_number]) . '" class="dropdown-item"><i class="fa fa-eye mr-2"></i>' . __('app.view') . '</a>';
            }

            if ($this->deleteComplaintPermission == 'all') {
                $action .=
                    '<a class="dropdown-item delete-table-row" href="javascript:;" data-ticket-id="' .
                    $row->id .
                    '">
                            <i class="fa fa-trash mr-2"></i>
                            ' .
                    trans('app.delete') .
                    '
                        </a>';
            }

            $action .= '</div>
                        </div>
                    </div>';

            return $action;
        });
        $datatables->addColumn('others', function ($row) {
            $others = '';

            if (!is_null($row->agent)) {
                $others .= '<div class="mb-2">' . __('modules.tickets.agent') . ': ' . (is_null($row->agent_id) ? '-' : $row->agent->name) . '</div> ';
            }

            $others .= '<div>' . __('modules.tasks.priority') . ': ' . __('app.' . $row->priority) . '</div> ';

            return $others;
        });

        $datatables->addColumn('status', function ($row) {
            if ($this->editComplaintPermission == 'all' || $this->editComplaintPermission == 'owned' && user()->id == $row->agent_id) {
                $status = '<select class="form-control select-picker change-status" data-ticket-id="' . $row->id . '">';
                $status .= '<option ';

                if ($row->status == 'open') {
                    $status .= 'selected';
                }

                $status .= '  data-content="<i class=\'fa fa-circle mr-2 text-red\'></i> ' . __('app.open') . '" value="open">' . __('app.open') . '</option>';
                $status .= '<option ';

                if ($row->status == 'pending') {
                    $status .= 'selected';
                }

                $status .= '  data-content="<i class=\'fa fa-circle mr-2 text-yellow\'></i> ' . __('app.pending') . '" value="pending">' . __('app.pending') . '</option>';
                $status .= '<option ';

                if ($row->status == 'resolved') {
                    $status .= 'selected';
                }

                $status .= '  data-content="<i class=\'fa fa-circle mr-2 text-dark-green\'></i> ' . __('app.resolved') . '" value="resolved">' . __('app.resolved') . '</option>';
                $status .= '<option ';

                if ($row->status == 'closed') {
                    $status .= 'selected';
                }

                $status .= '  data-content="<i class=\'fa fa-circle mr-2 text-blue\'></i> ' . __('app.closed') . '" value="closed">' . __('app.closed') . '</option>';

                $status .= '</select>';

                return $status;
            } else {
                if ($row->status == 'open') {
                    return '<i class="fa fa-circle mr-2 text-red"></i>' . __('app.open');
                } elseif ($row->status == 'pending') {
                    return '<i class="fa fa-circle mr-2 text-warning"></i>' . __('app.pending');
                } elseif ($row->status == 'resolved') {
                    return '<i class="fa fa-circle mr-2 text-dark-green"></i>' . __('app.resolved');
                } else {
                    return '<i class="fa fa-circle mr-2 text-blue"></i>' . __('app.closed');
                }
            }
            /* status end */
        });
        $datatables->editColumn('ticket_status', function ($row) {
            return $row->status;
        });
        $datatables->editColumn('subject', function ($row) {
            return '<a href="' . route('complaint.show', $row->complaint_number) . '" class="text-darkest-grey" >' . ucfirst($row->subject) . '</a>';
        });
        $datatables->addColumn('name', function ($row) {
            return $row->requester ? $row->requester->name : $row->complaint_number;
        });
        $datatables->editColumn('user_id', function ($row) {
            if (is_null($row->requester)) {
                return '';
            }

            if ($row->requester->hasRole('employee')) {
                return view('components.employee', [
                    'user' => $row->requester,
                ]);
            } else {
                return view('components.client', [
                    'user' => $row->requester,
                ]);
            }
        });
        $datatables->editColumn('updated_at', function ($row) {
            return $row->updated_at->timezone($this->company->timezone)->translatedFormat($this->company->date_format . ' ' . $this->company->time_format);
        });
        $datatables->setRowId(function ($row) {
            return 'row-' . $row->id;
        });
        $datatables->orderColumn('user_id', 'name $1');
        $datatables->orderColumn('status', 'id $1');

        $datatables->rawColumns(['others', 'action', 'subject', 'check', 'user_id', 'status']);
        $datatables->removeColumn('agent_id');
        $datatables->removeColumn('channel_id');
        $datatables->removeColumn('type_id');
        $datatables->removeColumn('deleted_at');

        // Custom Fields For export
        CustomField::customFieldData($datatables, Complaint::CUSTOM_FIELD_MODEL);

        return $datatables;
    }

    /**
     * @param Complaint $model
     * @return \Illuminate\Database\Eloquent\Builder
     */
    public function query(Complaint $model)
    {
        $request = $this->request();

        $model = $model
            ->with('requester', 'agent')
            ->select('complaint.*')
            ->join('users', 'users.id', '=', 'complaint.user_id');

        if ($this->viewComplaintPermission == 'owned') {
            $model->where(function ($query) {
                $query->where('complaint.user_id', '=', user()->id)
                ->orWhere('complaint.agent_id', '=', user()->id);
            });
        }

        if ($this->viewComplaintPermission == 'none') {
            $model->where(function ($query) {
                $query->where('complaint.user_id', '=', user()->id);
            });
        }

        if (!is_null($request->startDate) && $request->startDate != '') {
            $startDate = Carbon::createFromFormat($this->company->date_format, $request->startDate)->toDateString();
            $model->where(DB::raw('DATE(complaint.updated_at)'), '>=', $startDate);
        }

        if (!is_null($request->endDate) && $request->endDate != '') {
            $endDate = Carbon::createFromFormat($this->company->date_format, $request->endDate)->toDateString();
            $model->where(DB::raw('DATE(complaint.updated_at)'), '<=', $endDate);
        }

        if (!is_null($request->agentId) && $request->agentId != 'all' && $request->ticketFilterStatus != 'unassigned') {
            $model->where('complaint.agent_id', '=', $request->agentId);
        }

        if (!is_null($request->priority) && $request->priority != 'all') {
            $model->where('complaint.priority', '=', $request->priority);
        }

        if (!is_null($request->channelId) && $request->channelId != 'all') {
            $model->where('complaint.channel_id', '=', $request->channelId);
        }

        if (!is_null($request->typeId) && $request->typeId != 'all') {
            $model->where('complaint.type_id', '=', $request->typeId);
        }

        if (!is_null($request->tagId) && $request->tagId != 'all') {
            $model->join('ticket_tags', 'ticket_tags.ticket_id', 'complaint.id');
            $model->where('ticket_tags.tag_id', '=', $request->tagId);
        }

        if (!is_null($request->ticketStatus) && $request->ticketStatus != 'all' && $request->ticketFilterStatus == '') {
            $request->ticketStatus == 'unassigned' ? $model->whereNull('agent_id') : $model->where('complaint.status', '=', $request->ticketStatus);
        }

        if ($request->ticketFilterStatus != '') {
            $request->ticketFilterStatus == 'open' || $request->ticketFilterStatus == 'unassigned'
                ? $model->where(function ($query) {
                    $query->where('complaint.status', '=', 'open')->orWhere('complaint.status', '=', 'pending');
                })
                : $model->where(function ($query) {
                    $query->where('complaint.status', '=', 'resolved')->orWhere('complaint.status', '=', 'closed');
                });

            if ($request->ticketFilterStatus == 'unassigned') {
                $model->whereNull('agent_id');
            }
        }

        if ($request->searchText != '') {
            $model->where(function ($query) {
                $query
                    ->where('complaint.subject', 'like', '%' . request('searchText') . '%')
                    ->orWhere('complaint.complaint_number', 'like', '%' . request('searchText') . '%')
                    ->orWhere('complaint.status', 'like', '%' . request('searchText') . '%')
                    ->orWhere('users.name', 'like', '%' . request('searchText') . '%')
                    ->orWhere('complaint.priority', 'like', '%' . request('searchText') . '%');
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
        return $this->setBuilder('ticket-table', 5)
            ->parameters([
                'initComplete' => 'function () {
                    window.LaravelDataTables["ticket-table"].buttons().container()
                    .appendTo("#table-actions")
                }',
                'fnDrawCallback' => 'function( oSettings ) {
                    $("#ticket-table .select-picker").selectpicker();

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
        $data = [
            __('modules.tickets.ticket') . ' #' => ['data' => 'complaint_number', 'name' => 'complaint_number', 'title' => __('modules.tickets.ticket') . ' #', 'visible' => showId()],
            __('modules.tickets.ticketSubject') => ['data' => 'subject', 'name' => 'subject', 'title' => __('modules.tickets.ticketSubject'), 'width' => '20%'],
            __('app.name') => ['data' => 'name', 'name' => 'user_id', 'visible' => false, 'title' => __('app.name')],
            __('modules.tickets.requesterName') => ['data' => 'user_id', 'name' => 'user_id', 'visible' => !in_array('client', user_roles()), 'exportable' => false, 'title' => __('modules.tickets.requesterName'), 'width' => '20%'],
            __('modules.tickets.requestedOn') => ['data' => 'updated_at', 'name' => 'updated_at', 'title' => __('modules.tickets.requestedOn')],
            __('app.others') => ['data' => 'others', 'name' => 'others', 'sortable' => false, 'title' => __('app.others')],
            __('app.status') => ['data' => 'status', 'name' => 'status', 'exportable' => false, 'title' => __('app.status')],
            __('modules.ticketStatus') => ['data' => 'ticket_status', 'name' => 'ticket_status', 'visible' => false, 'title' => __('modules.ticketStatus')],
            Column::computed('action', __('app.action'))
                ->exportable(false)
                ->printable(false)
                ->orderable(false)
                ->searchable(false)
                ->addClass('text-right pr-20'),
        ];

        return array_merge($data, CustomFieldGroup::customFieldsDataMerge(new Complaint()));
    }
}
