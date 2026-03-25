<?php

namespace Modules\PMCore\app\Http\Controllers;

use App\ApiClasses\Error;
use App\ApiClasses\Success;
use App\Helpers\FormattingHelper;
use App\Http\Controllers\Controller;
use App\Services\AddonService\AddonService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Validator;
use Modules\PMCore\app\Enums\TimesheetStatus;
use Modules\PMCore\app\Models\Project;
use Modules\PMCore\app\Models\Timesheet;
use Yajra\DataTables\Facades\DataTables;

class TimesheetController extends Controller
{
    protected $addonService;

    public function __construct(AddonService $addonService)
    {
        $this->addonService = $addonService;

        // Apply resource authorization
        $this->authorizeResource(Timesheet::class, 'timesheet');
    }

    /**
     * Display a listing of timesheets
     */
    public function index()
    {
        $this->authorize('viewAny', Timesheet::class);

        return view('pmcore::timesheets.index');
    }

    /**
     * Get timesheets data for DataTables
     */
    public function indexAjax(Request $request)
    {
        $this->authorize('viewAny', Timesheet::class);

        $query = Timesheet::with(['user', 'project', 'task', 'approvedBy'])
            ->select('timesheets.*');

        // Restrict to own timesheets if user doesn't have view-timesheets permission
        if (! auth()->user()->can('pmcore.view-timesheets')) {
            $query->where('user_id', auth()->id());
        }

        // Apply filters
        if ($request->filled('user_id')) {
            $query->where('user_id', $request->user_id);
        }

        if ($request->filled('project_id')) {
            $query->where('project_id', $request->project_id);
        }

        if ($request->filled('status')) {
            $query->where('status', $request->status);
        }

        if ($request->filled('date_from')) {
            $query->whereDate('date', '>=', $request->date_from);
        }

        if ($request->filled('date_to')) {
            $query->whereDate('date', '<=', $request->date_to);
        }

        return DataTables::of($query)
            ->addColumn('user', function ($timesheet) {
                return view('components.datatable-user', ['user' => $timesheet->user])->render();
            })
            ->addColumn('project_name', function ($timesheet) {
                return $timesheet->project ? $timesheet->project->name : '-';
            })
            ->addColumn('task_name', function ($timesheet) {
                return $timesheet->task ? $timesheet->task->title : '-';
            })
            ->addColumn('formatted_date', function ($timesheet) {
                return FormattingHelper::formatDate($timesheet->date);
            })
            ->addColumn('formatted_hours', function ($timesheet) {
                return number_format($timesheet->hours, 2).' hrs';
            })
            ->addColumn('is_billable', function ($timesheet) {
                return $timesheet->is_billable
                    ? '<span class="badge bg-success">'.__('Yes').'</span>'
                    : '<span class="badge bg-secondary">'.__('No').'</span>';
            })
            ->addColumn('billing_amount', function ($timesheet) {
                if ($timesheet->is_billable && $timesheet->billing_rate) {
                    return FormattingHelper::formatCurrency($timesheet->hours * $timesheet->billing_rate);
                }

                return '-';
            })
            ->addColumn('status_badge', function ($timesheet) {
                $color = $timesheet->status->color();

                return '<span class="badge bg-'.$color.'">'.$timesheet->status->label().'</span>';
            })
            ->addColumn('approved_by', function ($timesheet) {
                if ($timesheet->approvedBy) {
                    return view('components.datatable-user', ['user' => $timesheet->approvedBy])->render();
                }

                return '-';
            })
            ->addColumn('actions', function ($timesheet) {
                $actions = [];

                // Only show edit for draft timesheets by the owner
                if ($timesheet->canBeEditedBy(Auth::user())) {
                    $actions[] = [
                        'label' => __('Edit'),
                        'icon' => 'bx bx-edit',
                        'url' => route('pmcore.timesheets.edit', $timesheet->id),
                    ];
                }

                // Only show submit for draft timesheets by the owner
                if ($timesheet->status === \Modules\PMCore\app\Enums\TimesheetStatus::DRAFT && $timesheet->user_id === Auth::id()) {
                    $actions[] = [
                        'label' => __('Submit'),
                        'icon' => 'bx bx-send',
                        'onclick' => "submitTimesheet({$timesheet->id})",
                    ];
                }

                // Only show approve/reject for submitted timesheets
                if ($timesheet->canBeApprovedBy(Auth::user())) {
                    $actions[] = [
                        'label' => __('Approve'),
                        'icon' => 'bx bx-check',
                        'onclick' => "approveTimesheet({$timesheet->id})",
                    ];
                    $actions[] = [
                        'label' => __('Reject'),
                        'icon' => 'bx bx-x',
                        'onclick' => "rejectTimesheet({$timesheet->id})",
                    ];
                }

                // Only show delete for draft timesheets by the owner
                if ($timesheet->status === \Modules\PMCore\app\Enums\TimesheetStatus::DRAFT && $timesheet->canBeEditedBy(Auth::user())) {
                    $actions[] = [
                        'label' => __('Delete'),
                        'icon' => 'bx bx-trash',
                        'onclick' => "deleteTimesheet({$timesheet->id})",
                    ];
                }

                return view('components.datatable-actions', [
                    'id' => $timesheet->id,
                    'actions' => $actions,
                ])->render();
            })
            ->rawColumns(['user', 'is_billable', 'status_badge', 'approved_by', 'actions'])
            ->make(true);
    }

    /**
     * Show the form for creating a new timesheet
     */
    public function create()
    {
        $projects = Project::active()->get();
        $tasks = collect(); // Will be loaded via AJAX based on project

        return view('pmcore::timesheets.create', compact('projects', 'tasks'));
    }

    /**
     * Store a newly created timesheet
     */
    public function store(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'user_id' => 'required|exists:users,id',
            'project_id' => 'required|exists:projects,id',
            'task_id' => 'nullable|numeric',
            'date' => 'required|date',
            'hours' => 'required|numeric|min:0.01|max:24',
            'description' => 'nullable|string|max:1000',
            'is_billable' => 'boolean',
            'billing_rate' => 'nullable|numeric|min:0',
            'cost_rate' => 'nullable|numeric|min:0',
        ]);

        if ($validator->fails()) {
            return Error::response([
                'message' => __('Validation failed'),
                'errors' => $validator->errors(),
            ]);
        }

        try {
            DB::beginTransaction();

            $timesheet = Timesheet::create($request->all());

            DB::commit();

            return Success::response([
                'message' => __('Timesheet created successfully!'),
                'timesheet' => $timesheet,
            ]);
        } catch (\Exception $e) {
            DB::rollBack();

            return Error::response(__('An error occurred while creating the timesheet.'));
        }
    }

    /**
     * Display the specified timesheet
     */
    public function show($id)
    {
        $timesheet = Timesheet::with(['user', 'project', 'task', 'approvedBy'])->findOrFail($id);

        return view('pmcore::timesheets.show', compact('timesheet'));
    }

    /**
     * Show the form for editing the specified timesheet
     */
    public function edit(Timesheet $timesheet)
    {
        $timesheet = Timesheet::findOrFail($id);

        if (! $timesheet->canBeEditedBy(Auth::user())) {
            return Error::response(__('You do not have permission to edit this timesheet.'));
        }
        $projects = Project::active()->get();
        // CRM tasks don't have direct project relationship, so we'll pass empty collection
        $tasks = collect();

        return view('pmcore::timesheets.edit', compact('timesheet', 'projects', 'tasks'));
    }

    /**
     * Update the specified timesheet
     */
    public function update(Request $request, Timesheet $timesheet)
    {
        $timesheet = Timesheet::findOrFail($id);

        if (! $timesheet->canBeEditedBy(Auth::user())) {
            return Error::response(__('You do not have permission to edit this timesheet.'));
        }

        $validator = Validator::make($request->all(), [
            'user_id' => 'required|exists:users,id',
            'project_id' => 'required|exists:projects,id',
            'task_id' => 'nullable|numeric',
            'date' => 'required|date',
            'hours' => 'required|numeric|min:0.01|max:24',
            'description' => 'nullable|string|max:1000',
            'is_billable' => 'boolean',
            'billing_rate' => 'nullable|numeric|min:0',
            'cost_rate' => 'nullable|numeric|min:0',
        ]);

        if ($validator->fails()) {
            return Error::response([
                'message' => __('Validation failed'),
                'errors' => $validator->errors(),
            ]);
        }

        try {
            DB::beginTransaction();

            $timesheet->update($request->all());

            DB::commit();

            return Success::response([
                'message' => __('Timesheet updated successfully!'),
                'timesheet' => $timesheet,
            ]);
        } catch (\Exception $e) {
            DB::rollBack();

            return Error::response(__('An error occurred while updating the timesheet.'));
        }
    }

    /**
     * Remove the specified timesheet
     */
    public function destroy(Timesheet $timesheet)
    {
        $timesheet = Timesheet::findOrFail($id);

        if (! $timesheet->canBeEditedBy(Auth::user())) {
            return Error::response(__('You do not have permission to delete this timesheet.'));
        }
        try {
            $timesheet->delete();

            return Success::response(['message' => __('Timesheet deleted successfully!')]);
        } catch (\Exception $e) {
            return Error::response(__('An error occurred while deleting the timesheet.'));
        }
    }

    /**
     * Approve a timesheet
     */
    public function approve(Timesheet $timesheet)
    {
        $timesheet = Timesheet::findOrFail($id);

        \Illuminate\Support\Facades\Log::debug('Approve timesheet attempt', [
            'timesheet_id' => $timesheet->id,
            'timesheet_status' => $timesheet->status->value,
            'user_id' => Auth::id(),
            'user_roles' => Auth::user()->getRoleNames()->toArray(),
            'is_project_manager' => $timesheet->project->project_manager_id === Auth::id(),
        ]);

        if (! $timesheet->canBeApprovedBy(Auth::user())) {
            return Error::response(__('You do not have permission to approve this timesheet.'));
        }

        try {
            $timesheet->approve(Auth::user());

            return Success::response(['message' => __('Timesheet approved successfully!')]);
        } catch (\Exception $e) {
            return Error::response(__('An error occurred while approving the timesheet.'));
        }
    }

    /**
     * Reject a timesheet
     */
    public function reject(Timesheet $timesheet)
    {
        if (! $timesheet->canBeApprovedBy(Auth::user())) {
            return Error::response(__('You do not have permission to reject this timesheet.'));
        }

        try {
            $timesheet->reject(Auth::user());

            return Success::response(['message' => __('Timesheet rejected successfully!')]);
        } catch (\Exception $e) {
            return Error::response(__('An error occurred while rejecting the timesheet.'));
        }
    }

    /**
     * Submit a timesheet for approval
     */
    public function submit(Timesheet $timesheet)
    {

        if ($timesheet->user_id !== Auth::id()) {
            return Error::response(__('You can only submit your own timesheets.'));
        }

        try {
            $timesheet->submit();

            return Success::response(['message' => __('Timesheet submitted for approval!')]);
        } catch (\Exception $e) {
            return Error::response(__('An error occurred while submitting the timesheet.'));
        }
    }

    /**
     * Get tasks for a specific project (AJAX)
     */
    public function getProjectTasks($projectId)
    {
        // CRM tasks use polymorphic relationship, not direct project_id
        // For now, return empty collection since tasks aren't directly linked to projects
        $tasks = collect();

        return Success::response(['tasks' => $tasks]);
    }

    /**
     * Get timesheet statistics
     */
    public function statistics(Request $request)
    {
        $query = Timesheet::query();

        // Apply filters
        if ($request->filled('user_id')) {
            $query->where('user_id', $request->user_id);
        }

        if ($request->filled('project_id')) {
            $query->where('project_id', $request->project_id);
        }

        if ($request->filled('date_from')) {
            $query->whereDate('date', '>=', $request->date_from);
        }

        if ($request->filled('date_to')) {
            $query->whereDate('date', '<=', $request->date_to);
        }

        $baseQuery = clone $query;

        $stats = [
            'total_hours' => $baseQuery->sum('hours'),
            'billable_hours' => (clone $query)->where('is_billable', true)->sum('hours'),
            'total_amount' => (clone $query)->whereNotNull('billing_rate')->sum(DB::raw('hours * billing_rate')),
            'approved_hours' => (clone $query)->where('status', TimesheetStatus::APPROVED)->sum('hours'),
            'pending_hours' => (clone $query)->where('status', TimesheetStatus::SUBMITTED)->sum('hours'),
            'draft_hours' => (clone $query)->where('status', TimesheetStatus::DRAFT)->sum('hours'),
        ];

        return Success::response(['statistics' => $stats]);
    }
}
