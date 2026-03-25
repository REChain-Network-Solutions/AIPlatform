<?php

namespace Modules\PMCore\app\Http\Controllers;

use App\Http\Controllers\Controller;
use App\Models\User;
use Carbon\Carbon;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Modules\PMCore\app\Models\Project;
use Modules\PMCore\app\Models\ResourceAllocation;
use Modules\PMCore\app\Models\Timesheet;

class ProjectReportController extends Controller
{
    /**
     * Display the reports index page.
     */
    public function index()
    {
        $stats = $this->getOverallStats();

        return view('pmcore::reports.index', compact('stats'));
    }

    /**
     * Display time tracking report.
     */
    public function timeReport(Request $request)
    {
        $startDate = $request->get('start_date', Carbon::now()->startOfMonth()->format('Y-m-d'));
        $endDate = $request->get('end_date', Carbon::now()->endOfMonth()->format('Y-m-d'));
        $projectId = $request->get('project_id');
        $userId = $request->get('user_id');

        $baseQuery = Timesheet::whereBetween('date', [$startDate, $endDate]);

        if ($projectId) {
            $baseQuery->where('project_id', $projectId);
        }

        if ($userId) {
            $baseQuery->where('user_id', $userId);
        }

        // Summary statistics - use clone to avoid modifying the base query
        $summary = [
            'total_hours' => (clone $baseQuery)->sum('hours'),
            'billable_hours' => (clone $baseQuery)->where('is_billable', true)->sum('hours'),
            'non_billable_hours' => (clone $baseQuery)->where('is_billable', false)->sum('hours'),
            'total_cost' => (clone $baseQuery)->sum('cost_amount'),
            'total_revenue' => (clone $baseQuery)->where('is_billable', true)->sum('billable_amount'),
        ];

        // Group by project
        $projectSummary = Timesheet::select(
            'project_id',
            DB::raw('SUM(hours) as total_hours'),
            DB::raw('SUM(CASE WHEN is_billable = 1 THEN hours ELSE 0 END) as billable_hours'),
            DB::raw('SUM(cost_amount) as total_cost'),
            DB::raw('SUM(CASE WHEN is_billable = 1 THEN billable_amount ELSE 0 END) as total_revenue')
        )
            ->with('project')
            ->whereBetween('date', [$startDate, $endDate])
            ->when($projectId, function ($q) use ($projectId) {
                $q->where('project_id', $projectId);
            })
            ->when($userId, function ($q) use ($userId) {
                $q->where('user_id', $userId);
            })
            ->groupBy('project_id')
            ->get();

        // Get available projects and users for filters
        $projects = Project::notArchived()->orderBy('name')->get();
        $users = User::whereHas('timesheets')->orderBy('name')->get();

        return view('pmcore::reports.time', compact(
            'summary',
            'projectSummary',
            'projects',
            'users',
            'startDate',
            'endDate',
            'projectId',
            'userId'
        ));
    }

    /**
     * Display budget report.
     */
    public function budgetReport(Request $request)
    {
        $status = $request->get('status');
        $type = $request->get('type');

        $query = Project::with(['client', 'projectManager'])
            ->notArchived();

        if ($status) {
            $query->where('status', $status);
        }

        if ($type) {
            $query->where('type', $type);
        }

        $projects = $query->get()->map(function ($project) {
            return [
                'project' => $project,
                'budget' => $project->budget,
                'actual_cost' => $project->actual_cost,
                'actual_revenue' => $project->actual_revenue,
                'budget_variance' => $project->budget_variance,
                'budget_variance_percentage' => $project->budget_variance_percentage,
                'profit_margin' => $project->profit_margin,
                'profit_margin_percentage' => $project->profit_margin_percentage,
                'is_over_budget' => $project->isOverBudget(),
                'total_hours' => $project->total_hours,
                'billable_hours' => $project->billable_hours,
            ];
        });

        // Summary statistics
        $summary = [
            'total_budget' => $projects->sum('budget'),
            'total_actual_cost' => $projects->sum('actual_cost'),
            'total_actual_revenue' => $projects->sum('actual_revenue'),
            'total_profit' => $projects->sum('profit_margin'),
            'projects_over_budget' => $projects->where('is_over_budget', true)->count(),
            'projects_on_track' => $projects->where('is_over_budget', false)->count(),
        ];

        return view('pmcore::reports.budget', compact('projects', 'summary', 'status', 'type'));
    }

    /**
     * Display resource utilization report.
     */
    public function resourceReport(Request $request)
    {
        $startDate = $request->get('start_date', Carbon::now()->startOfMonth()->format('Y-m-d'));
        $endDate = $request->get('end_date', Carbon::now()->endOfMonth()->format('Y-m-d'));
        $departmentId = $request->get('department_id');

        // Get all resources
        $query = User::with(['designation.department'])->whereHas('roles', function ($q) {
            $q->whereNotIn('name', ['client', 'customer']);
        });

        if ($departmentId) {
            $query->whereHas('designation', function ($q) use ($departmentId) {
                $q->where('department_id', $departmentId);
            });
        }

        $resources = $query->get()->map(function ($user) use ($startDate, $endDate) {
            $allocations = ResourceAllocation::where('user_id', $user->id)
                ->whereIn('status', ['active', 'planned'])
                ->where(function ($q) use ($startDate, $endDate) {
                    $q->whereBetween('start_date', [$startDate, $endDate])
                        ->orWhere(function ($q2) use ($startDate, $endDate) {
                            $q2->where('start_date', '<=', $startDate)
                                ->where(function ($q3) use ($endDate) {
                                    $q3->whereNull('end_date')
                                        ->orWhere('end_date', '>=', $endDate);
                                });
                        });
                })
                ->get();

            $timesheets = Timesheet::where('user_id', $user->id)
                ->whereBetween('date', [$startDate, $endDate])
                ->where('status', 'approved')
                ->get();

            $workingDays = $this->getWorkingDays($startDate, $endDate);
            $availableHours = $workingDays * 8;
            $allocatedHours = $allocations->sum(function ($allocation) use ($startDate, $endDate) {
                $allocationStart = Carbon::parse($allocation->start_date)->max($startDate);
                $allocationEnd = $allocation->end_date ? Carbon::parse($allocation->end_date)->min($endDate) : Carbon::parse($endDate);
                $days = $this->getWorkingDays($allocationStart, $allocationEnd);

                return $days * $allocation->hours_per_day * ($allocation->allocation_percentage / 100);
            });
            $actualHours = $timesheets->sum('hours');
            $utilizationPercentage = $availableHours > 0 ? round(($actualHours / $availableHours) * 100, 2) : 0;

            return [
                'user' => $user,
                'available_hours' => $availableHours,
                'allocated_hours' => $allocatedHours,
                'actual_hours' => $actualHours,
                'utilization_percentage' => $utilizationPercentage,
                'is_overallocated' => $allocatedHours > $availableHours,
                'is_underutilized' => $utilizationPercentage < 70,
            ];
        });

        // Summary statistics
        $summary = [
            'total_resources' => $resources->count(),
            'total_available_hours' => $resources->sum('available_hours'),
            'total_allocated_hours' => $resources->sum('allocated_hours'),
            'total_actual_hours' => $resources->sum('actual_hours'),
            'average_utilization' => $resources->avg('utilization_percentage'),
            'overallocated_resources' => $resources->where('is_overallocated', true)->count(),
            'underutilized_resources' => $resources->where('is_underutilized', true)->count(),
        ];

        // Get departments for filter
        $departments = [];
        if (class_exists('\Modules\HRCore\app\Models\Department')) {
            $departments = \Modules\HRCore\app\Models\Department::orderBy('name')->get();
        }

        return view('pmcore::reports.resource', compact(
            'resources',
            'summary',
            'departments',
            'startDate',
            'endDate',
            'departmentId'
        ));
    }

    /**
     * Get overall statistics for dashboard.
     */
    private function getOverallStats()
    {
        $activeProjects = Project::notArchived();

        return [
            'total_projects' => $activeProjects->count(),
            'ongoing_projects' => $activeProjects->ongoing()->count(),
            'completed_projects' => $activeProjects->completed()->count(),
            'total_budget' => $activeProjects->sum('budget'),
            'total_spent' => $activeProjects->sum('actual_cost'),
            'total_revenue' => $activeProjects->sum('actual_revenue'),
            'projects_over_budget' => $activeProjects->get()->filter->isOverBudget()->count(),
            'average_completion' => $activeProjects->avg('completion_percentage'),
        ];
    }

    /**
     * Get working days between two dates.
     */
    private function getWorkingDays($startDate, $endDate): int
    {
        $start = Carbon::parse($startDate);
        $end = Carbon::parse($endDate);
        $days = 0;

        while ($start <= $end) {
            if ($start->isWeekday()) {
                $days++;
            }
            $start->addDay();
        }

        return $days;
    }

    /**
     * Export report data.
     */
    public function export(Request $request)
    {
        $type = $request->get('type', 'time');
        $format = $request->get('format', 'csv');

        // This is a placeholder for export functionality
        // In a real implementation, you would generate CSV/Excel/PDF files

        return redirect()->back()->with('info', __('Export functionality will be implemented with DataImportExport addon.'));
    }
}
