<?php

namespace Modules\PMCore\app\Http\Controllers;

use App\Http\Controllers\Controller;
use App\Models\User;
use App\Services\Settings\ModuleSettingsService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Validator;
use Illuminate\Validation\Rule;
use Modules\PMCore\app\Enums\ProjectPriority;
use Modules\PMCore\app\Enums\ProjectStatus;
use Modules\PMCore\app\Enums\ProjectType;
use Modules\PMCore\app\Models\Project;
use Modules\PMCore\app\Services\PMIntegrationService;

class ProjectController extends Controller
{
    protected $integrationService;

    public function __construct(PMIntegrationService $integrationService)
    {
        $this->integrationService = $integrationService;

        // Apply resource authorization
        $this->authorizeResource(Project::class, 'project');
    }

    /**
     * Display a listing of projects.
     */
    public function index()
    {
        $this->authorize('viewAny', Project::class);

        $stats = $this->getProjectStats();
        $filters = $this->getFilterData();

        return view('pmcore::projects.index', compact('stats', 'filters'));
    }

    /**
     * Get project data for DataTables.
     */
    public function getDataAjax(Request $request)
    {
        $this->authorize('viewAny', Project::class);

        $query = Project::with(['client', 'projectManager', 'members'])
            ->select([
                'id',
                'name',
                'code',
                'status',
                'type',
                'priority',
                'start_date',
                'end_date',
                'budget',
                'client_id',
                'project_manager_id',
                'created_at',
            ]);

        // Apply filters
        if ($request->filled('status')) {
            $query->byStatus($request->status);
        }

        if ($request->filled('type')) {
            $query->byType($request->type);
        }

        if ($request->filled('priority')) {
            $query->where('priority', $request->priority);
        }

        if ($request->filled('project_manager_id')) {
            $query->managedBy($request->project_manager_id);
        }

        if ($request->filled('client_id')) {
            $query->where('client_id', $request->client_id);
        }

        return datatables($query)
            ->addColumn('actions', function ($project) {
                return view('pmcore::projects._partials._actions', compact('project'))->render();
            })
            ->addColumn('status_badge', function ($project) {
                return '<span class="badge bg-label-'.$project->status->color().'">'.
                  $project->status->label().'</span>';
            })
            ->addColumn('priority_badge', function ($project) {
                return '<span class="badge bg-label-'.$project->priority->color().'">'.
                  $project->priority->label().'</span>';
            })
            ->addColumn('progress', function ($project) {
                $percentage = $project->getProgressPercentage();

                return '<div class="progress" style="height: 6px;">
                          <div class="progress-bar" role="progressbar" style="width: '.$percentage.'%"></div>
                        </div>
                        <small class="text-muted">'.$percentage.'%</small>';
            })
            ->addColumn('client_name', function ($project) {
                return $project->client?->name ?? '-';
            })
            ->addColumn('manager_name', function ($project) {
                return $project->projectManager?->getFullName() ?? '-';
            })
            ->addColumn('members_count', function ($project) {
                return $project->activeMembers()->count();
            })
            ->editColumn('budget', function ($project) {
                return $project->budget ? number_format($project->budget, 2) : '-';
            })
            ->editColumn('start_date', function ($project) {
                return $project->start_date?->format('M d, Y') ?? '-';
            })
            ->editColumn('end_date', function ($project) {
                return $project->end_date?->format('M d, Y') ?? '-';
            })
            ->rawColumns(['actions', 'status_badge', 'priority_badge', 'progress'])
            ->make(true);
    }

    /**
     * Show the form for creating a new project.
     */
    public function create()
    {
        // For AJAX requests, return form data
        if (request()->ajax()) {
            $data = $this->getFormData();

            return \App\ApiClasses\Success::response($data);
        }

        // For full-page requests, return the create view
        return view('pmcore::projects.create');
    }

    /**
     * Store a newly created project.
     */
    public function store(Request $request)
    {
        $validator = $this->validateProject($request);

        if ($validator->fails()) {
            if ($request->ajax()) {
                return response()->json([
                    'status' => 'failed',
                    'statusCode' => 422,
                    'data' => 'Validation failed',
                    'errors' => $validator->errors()->toArray(),
                ], 422);
            }

            return redirect()->back()
                ->withErrors($validator)
                ->withInput();
        }

        try {
            $project = null;
            DB::transaction(function () use ($request, &$project) {
                // Get settings for default values
                $settingsService = app(ModuleSettingsService::class);
                $defaultStatus = $settingsService->get('PMCore', 'default_project_status', 'planning');
                $defaultPriority = $settingsService->get('PMCore', 'default_project_priority', 'medium');
                $defaultBillable = $settingsService->get('PMCore', 'default_is_billable', true);

                $project = Project::create([
                    'name' => $request->name,
                    'code' => $request->code,
                    'description' => $request->description,
                    'status' => $request->status ?? $defaultStatus,
                    'type' => $request->type ?? ProjectType::CLIENT->value,
                    'priority' => $request->priority ?? $defaultPriority,
                    'start_date' => $request->start_date,
                    'end_date' => $request->end_date,
                    'budget' => $request->budget,
                    'color_code' => $request->color_code ?? '#007bff',
                    'is_billable' => $request->boolean('is_billable', $defaultBillable),
                    'hourly_rate' => $request->hourly_rate,
                    'client_id' => $request->client_id,
                    'project_manager_id' => $request->project_manager_id,
                ]);

                // Add project manager as member if specified
                if ($request->project_manager_id) {
                    $project->addMember($request->project_manager_id, [
                        'role' => 'manager',
                        'hourly_rate' => $request->hourly_rate,
                    ]);
                }

                // Add additional members if specified
                if ($request->filled('member_ids')) {
                    foreach ($request->member_ids as $memberId) {
                        if ($memberId != $request->project_manager_id) {
                            $project->addMember($memberId, [
                                'role' => 'member',
                            ]);
                        }
                    }
                }
            });

            if ($request->ajax()) {
                return \App\ApiClasses\Success::response([
                    'message' => 'Project created successfully',
                    'project' => $project,
                ]);
            }

            return redirect()->route('pmcore.projects.index')
                ->with('success', 'Project created successfully.');

        } catch (\Exception $e) {
            if ($request->ajax()) {
                return \App\ApiClasses\Error::response('Failed to create project');
            }

            return redirect()->back()
                ->with('error', 'Failed to create project.')
                ->withInput();
        }
    }

    /**
     * Display the specified project.
     */
    public function show(Project $project)
    {
        $project->load([
            'client',
            'projectManager',
            'members.user',
            'activeMembers.user',
            'resourceAllocations' => function ($query) {
                $query->with('user')
                    ->whereIn('status', ['active', 'planned'])
                    ->orderBy('allocation_percentage', 'desc');
            },
        ]);

        // If AJAX request, return JSON data
        if (request()->ajax()) {
            return \App\ApiClasses\Success::response([
                'id' => $project->id,
                'name' => $project->name,
                'code' => $project->code,
                'description' => $project->description,
                'status' => $project->status->value,
                'type' => $project->type->value,
                'priority' => $project->priority->value,
                'start_date' => $project->start_date?->format('Y-m-d'),
                'end_date' => $project->end_date?->format('Y-m-d'),
                'budget' => $project->budget,
                'hourly_rate' => $project->hourly_rate,
                'color_code' => $project->color_code,
                'is_billable' => $project->is_billable,
                'client_id' => $project->client_id,
                'project_manager_id' => $project->project_manager_id,
                'client' => $project->client,
                'project_manager' => $project->projectManager,
                'members' => $project->activeMembers,
            ]);
        }

        $stats = [
            'total_members' => $project->activeMembers()->count(),
            'total_tasks' => $this->getProjectTaskCount($project),
            'completed_tasks' => $this->getProjectCompletedTaskCount($project),
            'pending_tasks' => $this->getProjectPendingTaskCount($project),
            'in_progress_tasks' => $this->getProjectInProgressTaskCount($project),
            'overdue_tasks' => $this->getProjectOverdueTaskCount($project),
            'milestones' => $this->getProjectMilestoneCount($project),
            'progress_percentage' => $project->getProgressPercentage(),
            'days_remaining' => $project->end_date ? now()->diffInDays($project->end_date, false) : null,
            'is_overdue' => $project->isOverdue(),
            'task_stats_by_status' => $this->getTaskStatsByStatus($project),
            'task_stats_by_priority' => $this->getTaskStatsByPriority($project),
            'recent_tasks' => $this->getRecentTasks($project),
            'upcoming_tasks' => $this->getUpcomingTasks($project),
            // Timesheet stats
            'total_hours' => $project->timesheets()->sum('hours'),
            'billable_hours' => $project->timesheets()->where('is_billable', true)->sum('hours'),
            'total_timesheet_amount' => $project->timesheets()->whereNotNull('billing_rate')->sum(DB::raw('hours * billing_rate')),
            'approved_hours' => $project->timesheets()->where('status', \Modules\PMCore\app\Enums\TimesheetStatus::APPROVED)->sum('hours'),
            // Resource allocation stats
            'total_allocated_resources' => $project->resourceAllocations()->whereIn('status', ['active', 'planned'])->distinct('user_id')->count('user_id'),
            'total_allocation_percentage' => $project->resourceAllocations()->whereIn('status', ['active', 'planned'])->sum('allocation_percentage'),
            'resource_hours_per_day' => $project->resourceAllocations()->whereIn('status', ['active', 'planned'])->sum('hours_per_day'),
        ];

        // Get recent timesheets
        $recent_timesheets = $project->timesheets()
            ->with(['user', 'task'])
            ->orderBy('date', 'desc')
            ->limit(5)
            ->get();

        // Get filter data for edit form dropdowns
        $filters = $this->getFilterData();

        return view('pmcore::projects.show', compact('project', 'stats', 'filters', 'recent_timesheets'));
    }

    /**
     * Show the form for editing the specified project.
     */
    public function edit(Project $project)
    {
        // For AJAX requests, return project data for the modal form
        if (request()->ajax()) {
            return \App\ApiClasses\Success::response([
                'id' => $project->id,
                'name' => $project->name,
                'code' => $project->code,
                'description' => $project->description,
                'status' => $project->status->value,
                'type' => $project->type->value,
                'priority' => $project->priority->value,
                'start_date' => $project->start_date?->format('Y-m-d'),
                'end_date' => $project->end_date?->format('Y-m-d'),
                'budget' => $project->budget,
                'hourly_rate' => $project->hourly_rate,
                'color_code' => $project->color_code,
                'is_billable' => $project->is_billable,
                'client_id' => $project->client_id,
                'project_manager_id' => $project->project_manager_id,
            ]);
        }

        // For full-page requests, return the edit view
        return view('pmcore::projects.edit', compact('project'));
    }

    /**
     * Update the specified project.
     */
    public function update(Request $request, Project $project)
    {
        // Debug: Log the incoming request data
        Log::info('Project update request data:', [
            'project_id' => $project->id,
            'request_data' => $request->all(),
        ]);

        $validator = $this->validateProject($request, $project->id);

        if ($validator->fails()) {
            Log::error('Project update validation failed:', [
                'project_id' => $project->id,
                'errors' => $validator->errors()->toArray(),
                'request_data' => $request->all(),
            ]);

            if ($request->ajax()) {
                return response()->json([
                    'status' => 'failed',
                    'statusCode' => 422,
                    'data' => 'Validation failed',
                    'errors' => $validator->errors()->toArray(),
                ], 422);
            }

            return redirect()->back()
                ->withErrors($validator)
                ->withInput();
        }

        try {
            DB::transaction(function () use ($request, $project) {
                $updateData = [
                    'name' => $request->name,
                    'code' => $request->code,
                    'description' => $request->description,
                    'status' => $request->status,
                    'type' => $request->type,
                    'priority' => $request->priority,
                    'start_date' => $request->start_date,
                    'end_date' => $request->end_date,
                    'budget' => $request->budget,
                    'color_code' => $request->color_code ?? '#007bff', // Provide default
                    'is_billable' => $request->boolean('is_billable'),
                    'hourly_rate' => $request->hourly_rate,
                    'client_id' => $request->client_id,
                    'project_manager_id' => $request->project_manager_id,
                ];

                Log::info('Project update data prepared:', [
                    'project_id' => $project->id,
                    'update_data' => $updateData,
                ]);

                $project->update($updateData);

                // Update project manager membership
                if ($request->project_manager_id) {
                    $existingManager = $project->members()
                        ->where('user_id', $request->project_manager_id)
                        ->first();

                    if ($existingManager) {
                        $existingManager->update(['role' => 'manager']);
                    } else {
                        $project->addMember($request->project_manager_id, [
                            'role' => 'manager',
                            'hourly_rate' => $request->hourly_rate,
                        ]);
                    }
                }
            });

            if ($request->ajax()) {
                return \App\ApiClasses\Success::response([
                    'message' => 'Project updated successfully',
                    'project' => $project,
                ]);
            }

            return redirect()->route('pmcore.projects.show', $project)
                ->with('success', 'Project updated successfully.');

        } catch (\Exception $e) {
            Log::error('Failed to update project: '.$e->getMessage(), [
                'project_id' => $project->id,
                'request_data' => $request->all(),
                'exception' => $e,
            ]);

            if ($request->ajax()) {
                return \App\ApiClasses\Error::response('Failed to update project: '.$e->getMessage());
            }

            return redirect()->back()
                ->with('error', 'Failed to update project.')
                ->withInput();
        }
    }

    /**
     * Remove the specified project.
     */
    public function destroy(Project $project)
    {
        try {
            DB::transaction(function () use ($project) {
                // Remove all project members
                $project->members()->delete();

                // If TaskSystem is available, handle task cleanup
                // Tasks would be soft deleted or reassigned
                // This depends on the TaskSystem implementation

                // Soft delete the project
                $project->delete();
            });

            if (request()->ajax()) {
                return \App\ApiClasses\Success::response('Project deleted successfully');
            }

            return redirect()->route('pmcore.projects.index')
                ->with('success', 'Project deleted successfully.');

        } catch (\Exception $e) {
            if (request()->ajax()) {
                return \App\ApiClasses\Error::response('Failed to delete project');
            }

            return redirect()->back()
                ->with('error', 'Failed to delete project.');
        }
    }

    /**
     * Add member to project.
     */
    public function addMember(Request $request, Project $project)
    {
        $this->authorize('addMember', $project);
        $validator = Validator::make($request->all(), [
            'user_id' => 'required|exists:users,id',
            'role' => 'required|string|in:member,lead,coordinator,manager',
            'hourly_rate' => 'nullable|numeric|min:0',
            'allocation_percentage' => 'nullable|integer|min:0|max:100',
        ]);

        if ($validator->fails()) {
            if ($request->ajax()) {
                return \App\ApiClasses\Error::response($validator->errors()->first());
            }

            return redirect()->back()->withErrors($validator)->withInput();
        }

        try {
            // Check if user is already a member
            if ($project->hasMember($request->user_id)) {
                if ($request->ajax()) {
                    return \App\ApiClasses\Error::response('User is already a member of this project');
                }

                return redirect()->back()->with('error', 'User is already a member of this project');
            }

            $member = $project->addMember($request->user_id, [
                'role' => $request->role,
                'hourly_rate' => $request->hourly_rate,
                'allocation_percentage' => $request->allocation_percentage ?? 100,
            ]);

            if ($request->ajax()) {
                return \App\ApiClasses\Success::response([
                    'message' => 'Member added successfully',
                    'member' => $member->load('user'),
                ]);
            }

            return redirect()->back()->with('success', 'Member added successfully');

        } catch (\Exception $e) {
            Log::error('Failed to add project member: '.$e->getMessage());

            if ($request->ajax()) {
                return \App\ApiClasses\Error::response('Failed to add member');
            }

            return redirect()->back()->with('error', 'Failed to add member');
        }
    }

    /**
     * Remove member from project.
     */
    public function removeMember(Request $request, Project $project, $memberId)
    {
        $this->authorize('removeMember', $project);
        try {
            $member = $project->members()->findOrFail($memberId);

            // Prevent removing project manager
            if ($member->role === 'manager' && $project->project_manager_id === $member->user_id) {
                if ($request->ajax()) {
                    return \App\ApiClasses\Error::response('Cannot remove project manager. Assign a new manager first.');
                }

                return redirect()->back()->with('error', 'Cannot remove project manager. Assign a new manager first.');
            }

            $member->update(['left_at' => now()]);

            if ($request->ajax()) {
                return \App\ApiClasses\Success::response([
                    'message' => 'Member removed successfully',
                ]);
            }

            return redirect()->back()->with('success', 'Member removed successfully');

        } catch (\Exception $e) {
            Log::error('Failed to remove project member: '.$e->getMessage());

            if ($request->ajax()) {
                return \App\ApiClasses\Error::response('Failed to remove member');
            }

            return redirect()->back()->with('error', 'Failed to remove member');
        }
    }

    /**
     * Update member role.
     */
    public function updateMemberRole(Request $request, Project $project, $memberId)
    {
        $this->authorize('manageTeam', $project);
        $validator = Validator::make($request->all(), [
            'role' => 'required|string|in:member,lead,coordinator,manager',
            'hourly_rate' => 'nullable|numeric|min:0',
            'allocation_percentage' => 'nullable|integer|min:0|max:100',
        ]);

        if ($validator->fails()) {
            if ($request->ajax()) {
                return \App\ApiClasses\Error::response($validator->errors()->first());
            }

            return redirect()->back()->withErrors($validator)->withInput();
        }

        try {
            $member = $project->members()->findOrFail($memberId);

            $updateData = [
                'role' => $request->role,
            ];

            // Only update hourly_rate if provided
            if ($request->filled('hourly_rate')) {
                $updateData['hourly_rate'] = $request->hourly_rate;
            }

            // Only update allocation_percentage if provided, otherwise keep current value
            if ($request->filled('allocation_percentage')) {
                $updateData['allocation_percentage'] = $request->allocation_percentage;
            }

            $member->update($updateData);

            if ($request->ajax()) {
                return \App\ApiClasses\Success::response([
                    'message' => 'Member role updated successfully',
                    'member' => $member->load('user'),
                ]);
            }

            return redirect()->back()->with('success', 'Member role updated successfully');

        } catch (\Exception $e) {
            Log::error('Failed to update member role: '.$e->getMessage());

            if ($request->ajax()) {
                return \App\ApiClasses\Error::response('Failed to update member role');
            }

            return redirect()->back()->with('error', 'Failed to update member role');
        }
    }

    /**
     * Search users for project member assignment.
     */
    public function searchUsers(Request $request)
    {
        $search = $request->get('search', '');
        $excludeRoles = $request->get('exclude_roles', []);

        $query = User::where('status', 'active');

        // Exclude users with specific roles if requested
        if (! empty($excludeRoles)) {
            $query->whereDoesntHave('roles', function ($q) use ($excludeRoles) {
                $q->whereIn('name', $excludeRoles);
            });
        }

        // Search by name or email
        if ($search) {
            $query->where(function ($q) use ($search) {
                $q->where('first_name', 'LIKE', "%{$search}%")
                    ->orWhere('last_name', 'LIKE', "%{$search}%")
                    ->orWhere('email', 'LIKE', "%{$search}%")
                    ->orWhereRaw("CONCAT(first_name, ' ', last_name) LIKE ?", ["%{$search}%"]);
            });
        }

        $users = $query->select('id', 'first_name', 'last_name', 'email')
            ->limit(20)
            ->get();

        $formattedUsers = $users->map(function ($user) {
            return [
                'id' => $user->id,
                'text' => $user->getFullName().' ('.$user->email.')',
                'name' => $user->getFullName(),
                'email' => $user->email,
            ];
        });

        return \App\ApiClasses\Success::response($formattedUsers);
    }

    /**
     * Update project financial values from timesheets.
     */
    protected function updateProjectFinancials($project): void
    {
        try {
            // Calculate actual cost from timesheets
            $actualCost = $project->timesheets()
                ->whereIn('status', ['approved', 'submitted'])
                ->sum(DB::raw('hours * COALESCE(cost_rate, hourly_rate, 0)'));

            // Calculate actual revenue from billable timesheets
            $actualRevenue = $project->timesheets()
                ->whereIn('status', ['approved', 'submitted'])
                ->where('is_billable', true)
                ->sum(DB::raw('hours * COALESCE(billing_rate, hourly_rate, 0)'));

            // Update the project's financial fields
            $project->update([
                'actual_cost' => $actualCost,
                'actual_revenue' => $actualRevenue,
            ]);
        } catch (\Exception $e) {
            Log::debug('Failed to update project financials: '.$e->getMessage());
        }
    }

    /**
     * Get project statistics.
     */
    protected function getProjectStats(): array
    {
        $activeProjects = Project::notArchived();

        // Calculate task counts across all projects
        $totalCompletedTasks = 0;
        $totalOverdueTasks = 0;
        $totalSpent = 0;
        $totalRevenue = 0;

        // Get all active projects for calculation
        $projects = $activeProjects->get();

        foreach ($projects as $project) {
            // Update actual cost and revenue from timesheets
            $this->updateProjectFinancials($project);

            // Calculate task counts
            $totalCompletedTasks += $this->getProjectCompletedTaskCount($project);
            $totalOverdueTasks += $this->getProjectOverdueTaskCount($project);

            // Get updated financial values
            $project->refresh();
            $totalSpent += $project->actual_cost ?? 0;
            $totalRevenue += $project->actual_revenue ?? 0;
        }

        return [
            'total_projects' => $projects->count(),
            'active_projects' => $activeProjects->ongoing()->count(),
            'completed_projects' => $activeProjects->completed()->count(),
            'overdue_projects' => $activeProjects->ongoing()
                ->whereNotNull('end_date')
                ->where('end_date', '<', now())
                ->count(),
            'total_budget' => $activeProjects->sum('budget'),
            'total_spent' => $totalSpent,
            'total_revenue' => $totalRevenue,
            'projects_over_budget' => $projects->filter->isOverBudget()->count(),
            'completed_tasks' => $totalCompletedTasks,
            'overdue_tasks' => $totalOverdueTasks,
        ];
    }

    /**
     * Get filter data for the index page.
     */
    protected function getFilterData(): array
    {
        $data = [
            'statuses' => ProjectStatus::cases(),
            'types' => ProjectType::cases(),
            'priorities' => ProjectPriority::cases(),
            'project_managers' => User::where('status', 'active')
                ->select('id', 'first_name', 'last_name')
                ->get(),
        ];

        // Add clients if CRMCore is available
        if ($this->integrationService->isCRMCoreAvailable()) {
            $data['clients'] = \Modules\CRMCore\app\Models\Company::select('id', 'name')
                ->limit(100)
                ->get();
        }

        return $data;
    }

    /**
     * Get member details for editing.
     */
    public function getMemberDetails(Request $request, Project $project, $memberId)
    {
        try {
            $member = $project->members()->with('user')->findOrFail($memberId);

            if ($request->ajax()) {
                return \App\ApiClasses\Success::response([
                    'member' => [
                        'id' => $member->id,
                        'user_id' => $member->user_id,
                        'user_name' => $member->user->getFullName(),
                        'role' => $member->role->value,
                        'hourly_rate' => $member->hourly_rate,
                        'allocation_percentage' => $member->allocation_percentage,
                    ],
                ]);
            }

            return response()->json(['error' => 'Invalid request'], 400);

        } catch (\Exception $e) {
            Log::error('Failed to get member details: '.$e->getMessage());

            if ($request->ajax()) {
                return \App\ApiClasses\Error::response('Failed to get member details');
            }

            return response()->json(['error' => 'Member not found'], 404);
        }
    }

    /**
     * Get filter data for project forms.
     */
    protected function getFormFilterData()
    {
        return [
            'statuses' => ProjectStatus::cases(),
            'types' => ProjectType::cases(),
            'priorities' => ProjectPriority::cases(),
            'project_managers' => $this->integrationService->getProjectManagers(),
            'clients' => $this->integrationService->getAvailableClients(),
        ];
    }

    /**
     * Get form data for create/edit views.
     */
    protected function getFormData(): array
    {
        return $this->getFilterData();
    }

    /**
     * Get project task count safely.
     */
    protected function getProjectTaskCount($project): int
    {
        try {
            if (class_exists('\\Modules\\CRMCore\\app\\Models\\Task')) {
                return \Modules\CRMCore\app\Models\Task::where('taskable_type', 'Modules\PMCore\app\Models\Project')
                    ->where('taskable_id', $project->id)
                    ->count();
            }
        } catch (\Exception $e) {
            Log::debug('Failed to get project task count: '.$e->getMessage());
        }

        return 0;
    }

    /**
     * Get project completed task count safely.
     */
    protected function getProjectCompletedTaskCount($project): int
    {
        try {
            if (class_exists('\\Modules\\CRMCore\\app\\Models\\Task')) {
                return \Modules\CRMCore\app\Models\Task::where('taskable_type', 'Modules\PMCore\app\Models\Project')
                    ->where('taskable_id', $project->id)
                    ->whereNotNull('completed_at')
                    ->count();
            }
        } catch (\Exception $e) {
            Log::debug('Failed to get project completed task count: '.$e->getMessage());
        }

        return 0;
    }

    /**
     * Get project pending task count safely.
     */
    protected function getProjectPendingTaskCount($project): int
    {
        try {
            if (class_exists('\\Modules\\CRMCore\\app\\Models\\Task')) {
                return $project->tasks()->whereNull('completed_at')->count();
            }
        } catch (\Exception $e) {
            Log::debug('Failed to get project pending task count: '.$e->getMessage());
        }

        return 0;
    }

    /**
     * Get project in progress task count safely.
     */
    protected function getProjectInProgressTaskCount($project): int
    {
        try {
            if (class_exists('\\Modules\\CRMCore\\app\\Models\\Task')) {
                return $project->tasks()
                    ->whereNotNull('time_started_at')
                    ->whereNull('completed_at')
                    ->count();
            }
        } catch (\Exception $e) {
            Log::debug('Failed to get project in progress task count: '.$e->getMessage());
        }

        return 0;
    }

    /**
     * Get project overdue task count safely.
     */
    protected function getProjectOverdueTaskCount($project): int
    {
        try {
            if (class_exists('\\Modules\\CRMCore\\app\\Models\\Task')) {
                return \Modules\CRMCore\app\Models\Task::where('taskable_type', 'Modules\PMCore\app\Models\Project')
                    ->where('taskable_id', $project->id)
                    ->where('due_date', '<', now())
                    ->whereNull('completed_at')
                    ->count();
            }
        } catch (\Exception $e) {
            Log::debug('Failed to get project overdue task count: '.$e->getMessage());
        }

        return 0;
    }

    /**
     * Get project milestone count safely.
     */
    protected function getProjectMilestoneCount($project): int
    {
        try {
            if (class_exists('\\Modules\\CRMCore\\app\\Models\\Task')) {
                return $project->tasks()->where('is_milestone', true)->count();
            }
        } catch (\Exception $e) {
            Log::debug('Failed to get project milestone count: '.$e->getMessage());
        }

        return 0;
    }

    /**
     * Get task statistics by status.
     */
    protected function getTaskStatsByStatus($project): array
    {
        try {
            if (class_exists('\\Modules\\CRMCore\\app\\Models\\Task') && class_exists('\\Modules\\CRMCore\\app\\Models\\TaskStatus')) {
                return $project->tasks()
                    ->join('task_statuses', 'crm_tasks.task_status_id', '=', 'task_statuses.id')
                    ->select('task_statuses.name', 'task_statuses.color', DB::raw('count(*) as count'))
                    ->groupBy('task_statuses.id', 'task_statuses.name', 'task_statuses.color')
                    ->get()
                    ->toArray();
            }
        } catch (\Exception $e) {
            Log::debug('Failed to get task stats by status: '.$e->getMessage());
        }

        return [];
    }

    /**
     * Get task statistics by priority.
     */
    protected function getTaskStatsByPriority($project): array
    {
        try {
            if (class_exists('\\Modules\\CRMCore\\app\\Models\\Task') && class_exists('\\Modules\\CRMCore\\app\\Models\\TaskPriority')) {
                return $project->tasks()
                    ->join('task_priorities', 'crm_tasks.task_priority_id', '=', 'task_priorities.id')
                    ->select('task_priorities.name', 'task_priorities.color', DB::raw('count(*) as count'))
                    ->groupBy('task_priorities.id', 'task_priorities.name', 'task_priorities.color')
                    ->get()
                    ->toArray();
            }
        } catch (\Exception $e) {
            Log::debug('Failed to get task stats by priority: '.$e->getMessage());
        }

        return [];
    }

    /**
     * Get recent tasks for the project.
     */
    protected function getRecentTasks($project, $limit = 5): array
    {
        try {
            if (class_exists('\\Modules\\CRMCore\\app\\Models\\Task')) {
                return $project->tasks()
                    ->with(['status', 'priority', 'assignedToUser'])
                    ->orderBy('created_at', 'desc')
                    ->limit($limit)
                    ->get()
                    ->map(function ($task) {
                        return [
                            'id' => $task->id,
                            'title' => $task->title,
                            'status' => $task->status?->name,
                            'status_color' => $task->status?->color,
                            'priority' => $task->priority?->name,
                            'priority_color' => $task->priority?->color,
                            'assigned_to' => $task->assignedToUser ? trim($task->assignedToUser->first_name.' '.$task->assignedToUser->last_name) : null,
                            'due_date' => $task->due_date?->format('Y-m-d'),
                            'completed_at' => $task->completed_at?->format('Y-m-d'),
                            'is_milestone' => $task->is_milestone,
                            'is_overdue' => $task->due_date && $task->due_date->isPast() && ! $task->completed_at,
                        ];
                    })
                    ->toArray();
            }
        } catch (\Exception $e) {
            Log::debug('Failed to get recent tasks: '.$e->getMessage());
        }

        return [];
    }

    /**
     * Get upcoming tasks for the project.
     */
    protected function getUpcomingTasks($project, $limit = 5): array
    {
        try {
            if (class_exists('\\Modules\\CRMCore\\app\\Models\\Task')) {
                return $project->tasks()
                    ->with(['status', 'priority', 'assignedToUser'])
                    ->whereNull('completed_at')
                    ->where('due_date', '>=', now())
                    ->orderBy('due_date', 'asc')
                    ->limit($limit)
                    ->get()
                    ->map(function ($task) {
                        return [
                            'id' => $task->id,
                            'title' => $task->title,
                            'status' => $task->status?->name,
                            'status_color' => $task->status?->color,
                            'priority' => $task->priority?->name,
                            'priority_color' => $task->priority?->color,
                            'assigned_to' => $task->assignedToUser ? trim($task->assignedToUser->first_name.' '.$task->assignedToUser->last_name) : null,
                            'due_date' => $task->due_date?->format('Y-m-d'),
                            'is_milestone' => $task->is_milestone,
                            'days_until_due' => $task->due_date ? now()->diffInDays($task->due_date, false) : null,
                        ];
                    })
                    ->toArray();
            }
        } catch (\Exception $e) {
            Log::debug('Failed to get upcoming tasks: '.$e->getMessage());
        }

        return [];
    }

    /**
     * Validate project data.
     */
    protected function validateProject(Request $request, $projectId = null)
    {
        $rules = [
            'name' => 'required|string|max:255',
            'code' => [
                'nullable',
                'string',
                'max:50',
                Rule::unique('projects', 'code')->ignore($projectId),
            ],
            'description' => 'nullable|string',
            'status' => ['required', Rule::in(array_column(ProjectStatus::cases(), 'value'))],
            'type' => ['required', Rule::in(array_column(ProjectType::cases(), 'value'))],
            'priority' => ['required', Rule::in(array_column(ProjectPriority::cases(), 'value'))],
            'start_date' => 'nullable|date',
            'end_date' => 'nullable|date|after_or_equal:start_date',
            'budget' => 'nullable|numeric|min:0',
            'color_code' => 'nullable|string|regex:/^#[a-fA-F0-9]{6}$/',
            'is_billable' => 'boolean',
            'hourly_rate' => 'nullable|numeric|min:0',
            'project_manager_id' => 'nullable|exists:users,id',
            'member_ids' => 'nullable|array',
            'member_ids.*' => 'exists:users,id',
        ];

        // Add client validation if CRMCore is available
        if ($this->integrationService->isCRMCoreAvailable()) {
            $rules['client_id'] = 'nullable|exists:companies,id';
        }

        return Validator::make($request->all(), $rules);
    }

    /**
     * Search for clients for project assignment.
     */
    public function searchClients(Request $request)
    {
        $search = $request->get('search');

        if (! $this->integrationService->isCRMCoreAvailable()) {
            return \App\ApiClasses\Success::response([]);
        }

        try {
            $companyModel = '\Modules\CRMCore\app\Models\Company';
            if (class_exists($companyModel)) {
                $query = $companyModel::query()
                    ->select('id', 'name', 'email_office')
                    ->orderBy('name');

                if ($search) {
                    $query->where(function ($q) use ($search) {
                        $q->where('name', 'like', '%'.$search.'%')
                            ->orWhere('email_office', 'like', '%'.$search.'%');
                    });
                }

                $companies = $query->limit(20)->get();

                $results = $companies->map(function ($company) {
                    return [
                        'id' => $company->id,
                        'text' => $company->name.($company->email_office ? ' ('.$company->email_office.')' : ''),
                    ];
                });

                return \App\ApiClasses\Success::response($results);
            }
        } catch (\Exception $e) {
            Log::error('Client search error: '.$e->getMessage());
        }

        return \App\ApiClasses\Success::response([]);
    }

    /**
     * Duplicate a project.
     */
    public function duplicate(Project $project)
    {
        try {
            DB::beginTransaction();

            // Create new project with duplicated data
            $newProject = Project::create([
                'name' => $project->name.' (Copy)',
                'code' => $project->code ? $project->code.'_COPY' : null,
                'description' => $project->description,
                'type' => $project->type,
                'status' => ProjectStatus::PLANNING, // Reset to planning
                'priority' => $project->priority,
                'client_id' => $project->client_id,
                'project_manager_id' => $project->project_manager_id,
                'start_date' => null, // Reset dates
                'end_date' => null,
                'budget' => $project->budget,
                'hourly_rate' => $project->hourly_rate,
                'color_code' => $project->color_code,
                'is_billable' => $project->is_billable,
                'created_by_id' => auth()->id(),
                'updated_by_id' => auth()->id(),
            ]);

            // Duplicate project members (excluding the project manager who is already set)
            foreach ($project->activeMembers as $member) {
                if ($member->user_id !== $project->project_manager_id) {
                    $newProject->members()->create([
                        'user_id' => $member->user_id,
                        'role' => $member->role,
                        'hourly_rate' => $member->hourly_rate,
                        'allocation_percentage' => $member->allocation_percentage,
                        'joined_at' => now(),
                        'created_by_id' => auth()->id(),
                        'updated_by_id' => auth()->id(),
                    ]);
                }
            }

            DB::commit();

            return \App\ApiClasses\Success::response([
                'message' => __('Project duplicated successfully!'),
                'project' => $newProject,
                'redirect' => route('pmcore.projects.show', $newProject->id),
            ]);

        } catch (\Exception $e) {
            DB::rollBack();
            Log::error('Project duplication failed: '.$e->getMessage());

            return \App\ApiClasses\Error::response(__('Failed to duplicate project. Please try again.'));
        }
    }

    /**
     * Archive a project.
     */
    public function archive(Project $project)
    {
        try {
            $project->update([
                'status' => ProjectStatus::COMPLETED,
                'updated_by_id' => auth()->id(),
            ]);

            return \App\ApiClasses\Success::response([
                'message' => __('Project archived successfully!'),
            ]);

        } catch (\Exception $e) {
            Log::error('Project archiving failed: '.$e->getMessage());

            return \App\ApiClasses\Error::response(__('Failed to archive project. Please try again.'));
        }
    }

    /**
     * Search projects for selection (AJAX).
     */
    public function searchProjects(Request $request)
    {
        $search = $request->get('search', '');

        $query = Project::select('id', 'name', 'code')
            ->orderBy('name');

        if ($search) {
            $query->where(function ($q) use ($search) {
                $q->where('name', 'like', '%'.$search.'%')
                    ->orWhere('code', 'like', '%'.$search.'%');
            });
        }

        $projects = $query->limit(20)->get();

        $results = $projects->map(function ($project) {
            return [
                'id' => $project->id,
                'text' => $project->name.' ('.$project->code.')',
            ];
        });

        return \App\ApiClasses\Success::response($results);
    }

    /**
     * Display project timesheets
     */
    public function timesheets(Project $project)
    {
        $this->authorize('view', $project);

        $stats = [
            'total_hours' => $project->timesheets()->sum('hours'),
            'billable_hours' => $project->timesheets()->where('is_billable', true)->sum('hours'),
            'total_amount' => $project->timesheets()->whereNotNull('billing_rate')->sum(DB::raw('hours * billing_rate')),
            'approved_hours' => $project->timesheets()->where('status', \Modules\PMCore\app\Enums\TimesheetStatus::APPROVED)->sum('hours'),
            'pending_hours' => $project->timesheets()->whereIn('status', [
                \Modules\PMCore\app\Enums\TimesheetStatus::DRAFT,
                \Modules\PMCore\app\Enums\TimesheetStatus::SUBMITTED,
            ])->sum('hours'),
            'users_count' => $project->timesheets()->distinct('user_id')->count('user_id'),
        ];

        return view('pmcore::projects.timesheets', compact('project', 'stats'));
    }
}
