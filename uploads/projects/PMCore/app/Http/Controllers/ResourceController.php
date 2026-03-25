<?php

namespace Modules\PMCore\app\Http\Controllers;

use App\Http\Controllers\Controller;
use App\Models\User;
use Carbon\Carbon;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;
use Modules\PMCore\app\Models\Project;
use Modules\PMCore\app\Models\ResourceAllocation;
use Modules\PMCore\app\Models\ResourceCapacity;
use Yajra\DataTables\Facades\DataTables;

class ResourceController extends Controller
{
    /**
     * Display resource allocation dashboard
     */
    public function index(Request $request)
    {
        $projectId = $request->get('project_id');
        $project = null;

        if ($projectId) {
            $project = Project::findOrFail($projectId);
        }

        $stats = [
            'total_resources' => User::whereHas('roles', function ($q) {
                $q->whereNotIn('name', ['client', 'customer']);
            })->count(),
            'allocated_resources' => ResourceAllocation::whereIn('status', ['active', 'planned'])
                ->when($projectId, function ($query) use ($projectId) {
                    $query->where('project_id', $projectId);
                })
                ->distinct('user_id')
                ->count('user_id'),
            'overallocated_resources' => $this->getOverallocatedResourcesCount($projectId),
            'available_resources' => $this->getAvailableResourcesCount($projectId),
        ];

        return view('pmcore::resources.index', compact('stats', 'project'));
    }

    /**
     * Get resources data for DataTable
     */
    public function indexAjax(Request $request)
    {
        $query = User::with(['roles'])
            ->whereHas('roles', function ($q) {
                $q->whereNotIn('name', ['client', 'customer']);
            })
            ->select('users.*');

        // Add filters
        if ($request->filled('department_id')) {
            $query->where('department_id', $request->department_id);
        }

        if ($request->filled('availability')) {
            // Filter by availability status
            if ($request->availability === 'available') {
                $query->whereDoesntHave('resourceAllocations', function ($q) {
                    $q->whereIn('status', ['active', 'planned'])
                        ->where('allocation_percentage', '>=', 100)
                        ->currentAndFuture();
                });
            } elseif ($request->availability === 'partial') {
                $query->whereHas('resourceAllocations', function ($q) {
                    $q->whereIn('status', ['active', 'planned'])
                        ->where('allocation_percentage', '<', 100)
                        ->currentAndFuture();
                });
            } elseif ($request->availability === 'unavailable') {
                $query->whereHas('resourceAllocations', function ($q) {
                    $q->whereIn('status', ['active', 'planned'])
                        ->where('allocation_percentage', '>=', 100)
                        ->currentAndFuture();
                });
            }
        }

        return DataTables::of($query)
            ->addColumn('user', function ($user) {
                return view('components.datatable-user', ['user' => $user])->render();
            })
            ->addColumn('role', function ($user) {
                return $user->roles->pluck('name')->implode(', ');
            })
            ->addColumn('current_allocation', function ($user) {
                $allocations = $user->resourceAllocations()
                    ->with('project')
                    ->whereIn('status', ['active', 'planned'])
                    ->currentAndFuture()
                    ->get();

                return view('pmcore::resources._allocation_summary', compact('allocations'))->render();
            })
            ->addColumn('availability', function ($user) {
                $totalAllocation = $user->resourceAllocations()
                    ->whereIn('status', ['active', 'planned'])
                    ->currentAndFuture()
                    ->sum('allocation_percentage');

                $available = 100 - $totalAllocation;
                $statusClass = $available > 50 ? 'success' : ($available > 0 ? 'warning' : 'danger');

                return '<div class="progress" style="height: 20px;">
                    <div class="progress-bar bg-'.$statusClass.'" style="width: '.$available.'%">
                        '.$available.'% Available
                    </div>
                </div>';
            })
            ->addColumn('actions', function ($user) {
                $userName = addslashes($user->name ?? $user->getFullName());

                return view('components.datatable-actions', [
                    'id' => $user->id,
                    'actions' => [
                        [
                            'label' => __('View Schedule'),
                            'icon' => 'bx bx-calendar',
                            'url' => route('pmcore.resources.schedule', $user->id),
                        ],
                        [
                            'label' => __('Allocate'),
                            'icon' => 'bx bx-plus',
                            'onclick' => "allocateResource({$user->id}, '{$userName}')",
                        ],
                    ],
                ])->render();
            })
            ->filterColumn('user', function ($query, $keyword) {
                $query->where(function ($q) use ($keyword) {
                    $q->where('users.name', 'like', "%{$keyword}%")
                        ->orWhere('users.email', 'like', "%{$keyword}%");

                    // Check if user has first_name and last_name fields for full name search
                    if (Schema::hasColumn('users', 'first_name')) {
                        $q->orWhere('users.first_name', 'like', "%{$keyword}%");
                    }
                    if (Schema::hasColumn('users', 'last_name')) {
                        $q->orWhere('users.last_name', 'like', "%{$keyword}%");
                    }
                });
            })
            ->filterColumn('role', function ($query, $keyword) {
                $query->whereHas('roles', function ($q) use ($keyword) {
                    $q->where('name', 'like', "%{$keyword}%");
                });
            })
            ->rawColumns(['user', 'current_allocation', 'availability', 'actions'])
            ->make(true);
    }

    /**
     * Show resource schedule/calendar
     */
    public function schedule($userId)
    {
        $user = User::findOrFail($userId);

        // Get allocations for the next 3 months
        $startDate = now()->startOfMonth();
        $endDate = now()->addMonths(3)->endOfMonth();

        $allocations = ResourceAllocation::with(['project', 'task'])
            ->where('user_id', $userId)
            ->whereIn('status', ['active', 'planned'])
            ->inDateRange($startDate, $endDate)
            ->orderBy('start_date')
            ->get();

        $capacities = ResourceCapacity::where('user_id', $userId)
            ->forDateRange($startDate, $endDate)
            ->get()
            ->keyBy('date');

        return view('pmcore::resources.schedule', compact('user', 'allocations', 'capacities', 'startDate', 'endDate'));
    }

    /**
     * Show form to create resource allocation
     */
    public function create(Request $request)
    {
        $user = null;
        $project = null;

        if ($request->filled('user_id')) {
            $user = User::findOrFail($request->user_id);
        }

        if ($request->filled('project_id')) {
            $project = Project::findOrFail($request->project_id);
        }

        return view('pmcore::resources.create', compact('user', 'project'));
    }

    /**
     * Store new resource allocation
     */
    public function store(Request $request)
    {
        $request->validate([
            'user_id' => 'required|exists:users,id',
            'project_id' => 'required|exists:projects,id',
            'start_date' => 'required|date',
            'end_date' => 'nullable|date|after_or_equal:start_date',
            'allocation_percentage' => 'required|numeric|min:0|max:100',
            'hours_per_day' => 'required|numeric|min:0.5|max:24',
            'allocation_type' => 'required|in:project,task,phase',
            'task_id' => 'nullable|required_if:allocation_type,task',
            'phase' => 'nullable|required_if:allocation_type,phase|string|max:255',
            'notes' => 'nullable|string',
            'is_billable' => 'boolean',
            'is_confirmed' => 'boolean',
        ]);

        DB::beginTransaction();
        try {
            $allocation = ResourceAllocation::create([
                'user_id' => $request->user_id,
                'project_id' => $request->project_id,
                'start_date' => $request->start_date,
                'end_date' => $request->end_date,
                'allocation_percentage' => $request->allocation_percentage,
                'hours_per_day' => $request->hours_per_day,
                'allocation_type' => $request->allocation_type,
                'task_id' => $request->task_id,
                'phase' => $request->phase,
                'notes' => $request->notes,
                'is_billable' => $request->boolean('is_billable', true),
                'is_confirmed' => $request->boolean('is_confirmed', false),
                'status' => $request->boolean('is_confirmed') ? 'active' : 'planned',
                'created_by_id' => Auth::id(),
                'updated_by_id' => Auth::id(),
            ]);

            // Check for capacity conflicts
            $conflicts = $allocation->checkCapacityConflicts();

            // Update resource capacity
            $this->updateResourceCapacity($allocation);

            DB::commit();

            if ($request->ajax()) {
                return \App\ApiClasses\Success::response([
                    'message' => __('Resource allocated successfully!'),
                    'allocation' => $allocation,
                    'conflicts' => $conflicts,
                ]);
            }

            if (count($conflicts) > 0) {
                return redirect()->route('pmcore.resources.index')
                    ->with('success', __('Resource allocated successfully!'))
                    ->with('warning', __('Warning: Resource is overallocated in some periods.'));
            }

            return redirect()->route('pmcore.resources.index')
                ->with('success', __('Resource allocated successfully!'));

        } catch (\Exception $e) {
            DB::rollback();

            if ($request->ajax()) {
                return \App\ApiClasses\Error::response(__('Failed to allocate resource.'));
            }

            return redirect()->back()
                ->withInput()
                ->with('error', __('Failed to allocate resource.'));
        }
    }

    /**
     * Show form to edit resource allocation
     */
    public function edit($id)
    {
        $allocation = ResourceAllocation::with(['user', 'project', 'task'])->findOrFail($id);

        if (! $allocation->canBeEditedBy(Auth::user())) {
            abort(403, 'Unauthorized');
        }

        return view('pmcore::resources.edit', compact('allocation'));
    }

    /**
     * Update resource allocation
     */
    public function update(Request $request, $id)
    {
        $allocation = ResourceAllocation::findOrFail($id);

        if (! $allocation->canBeEditedBy(Auth::user())) {
            if ($request->ajax()) {
                return \App\ApiClasses\Error::response(__('You do not have permission to edit this allocation.'));
            }
            abort(403);
        }

        $request->validate([
            'start_date' => 'required|date',
            'end_date' => 'nullable|date|after_or_equal:start_date',
            'allocation_percentage' => 'required|numeric|min:0|max:100',
            'hours_per_day' => 'required|numeric|min:0.5|max:24',
            'notes' => 'nullable|string',
            'is_billable' => 'boolean',
            'is_confirmed' => 'boolean',
        ]);

        DB::beginTransaction();
        try {
            $allocation->update([
                'start_date' => $request->start_date,
                'end_date' => $request->end_date,
                'allocation_percentage' => $request->allocation_percentage,
                'hours_per_day' => $request->hours_per_day,
                'notes' => $request->notes,
                'is_billable' => $request->boolean('is_billable', true),
                'is_confirmed' => $request->boolean('is_confirmed', false),
                'status' => $request->boolean('is_confirmed') ? 'active' : $allocation->status,
                'updated_by_id' => Auth::id(),
            ]);

            // Check for capacity conflicts
            $conflicts = $allocation->checkCapacityConflicts();

            // Update resource capacity
            $this->updateResourceCapacity($allocation);

            DB::commit();

            if ($request->ajax()) {
                return \App\ApiClasses\Success::response([
                    'message' => __('Resource allocation updated successfully!'),
                    'allocation' => $allocation,
                    'conflicts' => $conflicts,
                ]);
            }

            return redirect()->route('pmcore.resources.index')
                ->with('success', __('Resource allocation updated successfully!'));

        } catch (\Exception $e) {
            DB::rollback();

            if ($request->ajax()) {
                return \App\ApiClasses\Error::response(__('Failed to update resource allocation.'));
            }

            return redirect()->back()
                ->withInput()
                ->with('error', __('Failed to update resource allocation.'));
        }
    }

    /**
     * Delete resource allocation
     */
    public function destroy(Request $request, $id)
    {
        $allocation = ResourceAllocation::findOrFail($id);

        if (! $allocation->canBeEditedBy(Auth::user())) {
            if ($request->ajax()) {
                return \App\ApiClasses\Error::response(__('You do not have permission to delete this allocation.'));
            }
            abort(403);
        }

        try {
            $userId = $allocation->user_id;
            $dates = $this->getDateRange($allocation->start_date, $allocation->end_date);

            $allocation->delete();

            // Update capacity for affected dates
            foreach ($dates as $date) {
                ResourceCapacity::updateAllocatedHours($userId, $date);
            }

            if ($request->ajax()) {
                return \App\ApiClasses\Success::response(__('Resource allocation deleted successfully!'));
            }

            return redirect()->route('pmcore.resources.index')
                ->with('success', __('Resource allocation deleted successfully!'));

        } catch (\Exception $e) {
            if ($request->ajax()) {
                return \App\ApiClasses\Error::response(__('Failed to delete resource allocation.'));
            }

            return redirect()->back()
                ->with('error', __('Failed to delete resource allocation.'));
        }
    }

    /**
     * Get resource availability for a date range
     */
    public function availability(Request $request)
    {
        $request->validate([
            'user_id' => 'required|exists:users,id',
            'start_date' => 'required|date',
            'end_date' => 'required|date|after_or_equal:start_date',
        ]);

        $userId = $request->user_id;
        $startDate = Carbon::parse($request->start_date);
        $endDate = Carbon::parse($request->end_date);

        // Generate capacity records if they don't exist
        ResourceCapacity::generateForUser($userId, $startDate, $endDate);

        $capacities = ResourceCapacity::where('user_id', $userId)
            ->forDateRange($startDate, $endDate)
            ->get();

        $availability = [];
        foreach ($capacities as $capacity) {
            $availability[] = [
                'date' => $capacity->date->format('Y-m-d'),
                'available_hours' => $capacity->available_hours,
                'allocated_hours' => $capacity->allocated_hours,
                'remaining_hours' => $capacity->remaining_hours,
                'is_working_day' => $capacity->is_working_day,
                'is_overallocated' => $capacity->is_overallocated,
                'allocation_percentage' => $capacity->allocation_percentage,
            ];
        }

        return \App\ApiClasses\Success::response([
            'availability' => $availability,
        ]);
    }

    /**
     * Update resource capacity for an allocation
     */
    private function updateResourceCapacity(ResourceAllocation $allocation): void
    {
        $dates = $this->getDateRange($allocation->start_date, $allocation->end_date);

        foreach ($dates as $date) {
            ResourceCapacity::updateAllocatedHours($allocation->user_id, $date);
        }
    }

    /**
     * Get date range for allocation
     */
    private function getDateRange(Carbon $startDate, ?Carbon $endDate): array
    {
        $dates = [];
        $current = $startDate->copy();
        $end = $endDate ?? now()->addYear();

        while ($current <= $end) {
            if ($current->isWeekday()) {
                $dates[] = $current->copy();
            }
            $current->addDay();
        }

        return $dates;
    }

    /**
     * Get count of overallocated resources
     */
    private function getOverallocatedResourcesCount($projectId = null): int
    {
        $query = ResourceAllocation::whereIn('status', ['active', 'planned'])
            ->currentAndFuture();

        if ($projectId) {
            // For project-specific view, get users overallocated on this project
            $query->where('project_id', $projectId);
        }

        $overallocatedUserIds = $query
            ->select('user_id')
            ->selectRaw('SUM(allocation_percentage) as total_percentage')
            ->groupBy('user_id')
            ->having('total_percentage', '>', 100)
            ->pluck('user_id');

        return $overallocatedUserIds->count();
    }

    /**
     * Get count of available resources
     */
    private function getAvailableResourcesCount($projectId = null): int
    {
        // Get all users who are not clients/customers
        $allResourceUsers = User::whereHas('roles', function ($q) {
            $q->whereNotIn('name', ['client', 'customer']);
        })->pluck('id');

        // Get users who are fully allocated (100% or more)
        $query = ResourceAllocation::whereIn('status', ['active', 'planned'])
            ->currentAndFuture();

        if ($projectId) {
            // For project-specific view, show users available for this project
            $query->where('project_id', $projectId);
        }

        $fullyAllocatedUserIds = $query
            ->select('user_id')
            ->selectRaw('SUM(allocation_percentage) as total_percentage')
            ->groupBy('user_id')
            ->having('total_percentage', '>=', 100)
            ->pluck('user_id');

        // Count available users (those not fully allocated)
        return $allResourceUsers->diff($fullyAllocatedUserIds)->count();
    }

    /**
     * Display capacity planning dashboard
     */
    public function capacity(Request $request)
    {
        $startDate = $request->get('start_date', now()->startOfMonth()->format('Y-m-d'));
        $endDate = $request->get('end_date', now()->endOfMonth()->format('Y-m-d'));
        $departmentId = $request->get('department_id');

        // Get all resources
        $query = User::whereHas('roles', function ($q) {
            $q->whereNotIn('name', ['client', 'customer']);
        });

        if ($departmentId) {
            $query->where('department_id', $departmentId);
        }

        $resources = $query->get();

        // Calculate capacity data
        $capacityData = $this->calculateCapacityData($resources, $startDate, $endDate);

        // Get departments for filter
        $departments = [];
        if (class_exists('\Modules\HRCore\app\Models\Department')) {
            $departments = \Modules\HRCore\app\Models\Department::orderBy('name')->get();
        }

        return view('pmcore::resources.capacity', compact('capacityData', 'startDate', 'endDate', 'departments'));
    }

    /**
     * Get capacity data for charts (AJAX)
     */
    public function capacityData(Request $request)
    {
        $startDate = Carbon::parse($request->get('start_date', now()->startOfMonth()));
        $endDate = Carbon::parse($request->get('end_date', now()->endOfMonth()));
        $departmentId = $request->get('department_id');
        $viewType = $request->get('view_type', 'utilization'); // utilization, forecast, heatmap

        // Get resources
        $query = User::whereHas('roles', function ($q) {
            $q->whereNotIn('name', ['client', 'customer']);
        });

        if ($departmentId) {
            $query->where('department_id', $departmentId);
        }

        $resources = $query->get();

        switch ($viewType) {
            case 'utilization':
                $data = $this->getUtilizationData($resources, $startDate, $endDate);
                break;
            case 'forecast':
                $data = $this->getForecastData($resources, $startDate, $endDate);
                break;
            case 'heatmap':
                $data = $this->getHeatmapData($resources, $startDate, $endDate);
                break;
            default:
                $data = [];
        }

        return \App\ApiClasses\Success::response($data);
    }

    /**
     * Calculate capacity data for resources
     */
    private function calculateCapacityData($resources, $startDate, $endDate): array
    {
        $startDate = Carbon::parse($startDate);
        $endDate = Carbon::parse($endDate);

        $data = [
            'total_resources' => $resources->count(),
            'total_capacity_hours' => 0,
            'allocated_hours' => 0,
            'available_hours' => 0,
            'utilization_percentage' => 0,
            'overallocated_resources' => 0,
            'underutilized_resources' => 0,
            'resource_breakdown' => [],
        ];

        foreach ($resources as $resource) {
            $allocations = ResourceAllocation::where('user_id', $resource->id)
                ->whereIn('status', ['active', 'planned'])
                ->inDateRange($startDate, $endDate)
                ->get();

            $workingDays = $this->getWorkingDays($startDate, $endDate);
            $totalCapacity = $workingDays * 8; // 8 hours per day
            $allocatedHours = 0;

            foreach ($allocations as $allocation) {
                $allocationStart = Carbon::parse($allocation->start_date)->max($startDate);
                $allocationEnd = $allocation->end_date ? Carbon::parse($allocation->end_date)->min($endDate) : $endDate;
                $allocationDays = $this->getWorkingDays($allocationStart, $allocationEnd);
                $allocatedHours += ($allocationDays * $allocation->hours_per_day * ($allocation->allocation_percentage / 100));
            }

            $utilization = $totalCapacity > 0 ? ($allocatedHours / $totalCapacity) * 100 : 0;

            $data['total_capacity_hours'] += $totalCapacity;
            $data['allocated_hours'] += $allocatedHours;

            if ($utilization > 100) {
                $data['overallocated_resources']++;
            } elseif ($utilization < 70) {
                $data['underutilized_resources']++;
            }

            $data['resource_breakdown'][] = [
                'resource' => $resource,
                'capacity_hours' => $totalCapacity,
                'allocated_hours' => $allocatedHours,
                'available_hours' => max(0, $totalCapacity - $allocatedHours),
                'utilization_percentage' => round($utilization, 1),
                'is_overallocated' => $utilization > 100,
                'is_underutilized' => $utilization < 70,
            ];
        }

        $data['available_hours'] = max(0, $data['total_capacity_hours'] - $data['allocated_hours']);
        $data['utilization_percentage'] = $data['total_capacity_hours'] > 0
            ? round(($data['allocated_hours'] / $data['total_capacity_hours']) * 100, 1)
            : 0;

        return $data;
    }

    /**
     * Get utilization data for charts
     */
    private function getUtilizationData($resources, Carbon $startDate, Carbon $endDate): array
    {
        $data = [
            'categories' => [],
            'series' => [
                ['name' => 'Allocated %', 'data' => []],
                ['name' => 'Available %', 'data' => []],
            ],
        ];

        foreach ($resources as $resource) {
            $workingDays = $this->getWorkingDays($startDate, $endDate);
            $totalCapacity = $workingDays * 8;

            $allocations = ResourceAllocation::where('user_id', $resource->id)
                ->whereIn('status', ['active', 'planned'])
                ->inDateRange($startDate, $endDate)
                ->get();

            $allocatedHours = 0;
            foreach ($allocations as $allocation) {
                $allocationStart = Carbon::parse($allocation->start_date)->max($startDate);
                $allocationEnd = $allocation->end_date ? Carbon::parse($allocation->end_date)->min($endDate) : $endDate;
                $allocationDays = $this->getWorkingDays($allocationStart, $allocationEnd);
                $allocatedHours += ($allocationDays * $allocation->hours_per_day * ($allocation->allocation_percentage / 100));
            }

            $utilization = $totalCapacity > 0 ? ($allocatedHours / $totalCapacity) * 100 : 0;

            $data['categories'][] = $resource->name;
            $data['series'][0]['data'][] = round(min($utilization, 100), 1);
            $data['series'][1]['data'][] = round(max(0, 100 - $utilization), 1);
        }

        return $data;
    }

    /**
     * Get forecast data
     */
    private function getForecastData($resources, Carbon $startDate, Carbon $endDate): array
    {
        $data = [
            'categories' => [],
            'series' => [
                ['name' => 'Current Allocation', 'data' => []],
                ['name' => 'Planned Allocation', 'data' => []],
                ['name' => 'Forecasted Demand', 'data' => []],
            ],
        ];

        // Generate monthly data points
        $current = $startDate->copy()->startOfMonth();
        while ($current <= $endDate) {
            $monthEnd = $current->copy()->endOfMonth();

            $currentAllocation = 0;
            $plannedAllocation = 0;

            foreach ($resources as $resource) {
                $allocations = ResourceAllocation::where('user_id', $resource->id)
                    ->inDateRange($current, $monthEnd)
                    ->get();

                foreach ($allocations as $allocation) {
                    if ($allocation->status === 'active') {
                        $currentAllocation += $allocation->allocation_percentage;
                    } else {
                        $plannedAllocation += $allocation->allocation_percentage;
                    }
                }
            }

            $totalResources = $resources->count() * 100; // 100% per resource
            $currentUtilization = $totalResources > 0 ? ($currentAllocation / $totalResources) * 100 : 0;
            $plannedUtilization = $totalResources > 0 ? ($plannedAllocation / $totalResources) * 100 : 0;

            // Simple forecast based on trend (you can make this more sophisticated)
            $forecastedDemand = $currentUtilization * 1.1; // 10% growth assumption

            $data['categories'][] = $current->format('M Y');
            $data['series'][0]['data'][] = round($currentUtilization, 1);
            $data['series'][1]['data'][] = round($plannedUtilization, 1);
            $data['series'][2]['data'][] = round($forecastedDemand, 1);

            $current->addMonth();
        }

        return $data;
    }

    /**
     * Get heatmap data
     */
    private function getHeatmapData($resources, Carbon $startDate, Carbon $endDate): array
    {
        $data = [
            'resources' => [],
            'dates' => [],
            'allocations' => [],
        ];

        // Generate date range
        $current = $startDate->copy();
        while ($current <= $endDate) {
            if ($current->isWeekday()) {
                $data['dates'][] = $current->format('Y-m-d');
            }
            $current->addDay();
        }

        // Get allocation data for each resource
        foreach ($resources as $resource) {
            $data['resources'][] = $resource->name;
            $resourceAllocations = [];

            foreach ($data['dates'] as $date) {
                $dateCarbon = Carbon::parse($date);
                $dayAllocation = ResourceAllocation::where('user_id', $resource->id)
                    ->whereIn('status', ['active', 'planned'])
                    ->where('start_date', '<=', $date)
                    ->where(function ($q) use ($date) {
                        $q->whereNull('end_date')
                            ->orWhere('end_date', '>=', $date);
                    })
                    ->sum('allocation_percentage');

                $resourceAllocations[] = min($dayAllocation, 100);
            }

            $data['allocations'][] = $resourceAllocations;
        }

        return $data;
    }

    /**
     * Get working days between two dates
     */
    private function getWorkingDays(Carbon $startDate, Carbon $endDate): int
    {
        $days = 0;
        $current = $startDate->copy();

        while ($current <= $endDate) {
            if ($current->isWeekday()) {
                $days++;
            }
            $current->addDay();
        }

        return $days;
    }
}
