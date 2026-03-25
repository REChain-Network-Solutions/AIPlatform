<?php

namespace Modules\PMCore\app\Http\Controllers;

use App\ApiClasses\Error;
use App\ApiClasses\Success;
use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Validator;
use Modules\CRMCore\app\Models\Task;
use Modules\CRMCore\app\Models\TaskPriority;
use Modules\CRMCore\app\Models\TaskStatus;
use Modules\PMCore\app\Models\Project;

class ProjectTaskController extends Controller
{
    public function index(Request $request, Project $project)
    {
        $statuses = TaskStatus::all();
        $priorities = TaskPriority::all();

        // Get only project members for task assignment
        $projectMembers = $project->members()->with('user')->get();
        $users = $projectMembers->map(function ($member) {
            return [
                'id' => $member->user->id,
                'name' => trim($member->user->first_name.' '.$member->user->last_name),
                'first_name' => $member->user->first_name,
                'last_name' => $member->user->last_name,
            ];
        });

        return view('pmcore::projects.tasks.index', compact('project', 'statuses', 'priorities', 'users'));
    }

    public function board(Request $request, Project $project)
    {
        $statuses = TaskStatus::all();
        $priorities = TaskPriority::all();

        // Get only project members for task assignment
        $projectMembers = $project->members()->with('user')->get();
        $users = $projectMembers->map(function ($member) {
            return [
                'id' => $member->user->id,
                'name' => trim($member->user->first_name.' '.$member->user->last_name),
                'first_name' => $member->user->first_name,
                'last_name' => $member->user->last_name,
            ];
        });

        return view('pmcore::projects.tasks.board', compact('project', 'statuses', 'priorities', 'users'));
    }

    public function getDataAjax(Request $request, Project $project)
    {
        $query = $project->tasks()->with(['status', 'priority', 'assignedToUser']);

        // Apply filters
        if ($request->has('task_status_id') && $request->task_status_id) {
            $query->where('task_status_id', $request->task_status_id);
        }

        if ($request->has('task_priority_id') && $request->task_priority_id) {
            $query->where('task_priority_id', $request->task_priority_id);
        }

        if ($request->has('assigned_to_user_id') && $request->assigned_to_user_id) {
            $query->where('assigned_to_user_id', $request->assigned_to_user_id);
        }

        // Handle board view request
        if ($request->has('board_view') && $request->board_view) {
            $tasks = $query->get();

            // Transform tasks for board view
            $transformedTasks = $tasks->map(function ($task) {
                return [
                    'id' => $task->id,
                    'title' => $task->title,
                    'description' => $task->description,
                    'task_status_id' => $task->task_status_id,
                    'task_priority_id' => $task->task_priority_id,
                    'assigned_to_user_id' => $task->assigned_to_user_id,
                    'due_date' => $task->due_date ? $task->due_date->format('Y-m-d') : null,
                    'estimated_hours' => $task->estimated_hours,
                    'is_milestone' => $task->is_milestone,
                    'completed_at' => $task->completed_at,
                    'time_started_at' => $task->time_started_at,
                    'status' => $task->status ? ['id' => $task->status->id, 'name' => $task->status->name] : null,
                    'priority' => $task->priority ? ['id' => $task->priority->id, 'name' => $task->priority->name] : null,
                    'assigned_to_user' => $task->assignedToUser ? [
                        'id' => $task->assignedToUser->id,
                        'first_name' => $task->assignedToUser->first_name,
                        'last_name' => $task->assignedToUser->last_name,
                    ] : null,
                ];
            });

            return Success::response($transformedTasks);
        }

        $tasks = $query->get();

        return datatables($tasks)
            ->addColumn('status', function ($task) {
                $statusHtml = $task->status ? $task->status->name : '-';

                // Add completion indicator
                if ($task->completed_at) {
                    $statusHtml .= ' <span class="badge bg-success ms-1">Completed</span>';
                } elseif ($task->time_started_at) {
                    $statusHtml .= ' <span class="badge bg-warning ms-1">Running</span>';
                }

                return $statusHtml;
            })
            ->addColumn('priority', function ($task) {
                return $task->priority ? $task->priority->name : '-';
            })
            ->addColumn('assigned_to', function ($task) {
                if ($task->assignedToUser) {
                    return trim($task->assignedToUser->first_name.' '.$task->assignedToUser->last_name);
                }

                return __('Unassigned');
            })
            ->addColumn('due_date', function ($task) {
                return $task->due_date ? $task->due_date->format('Y-m-d') : '-';
            })
            ->addColumn('actions', function ($task) {
                return view('pmcore::projects.tasks._actions', compact('task'));
            })
            ->rawColumns(['status', 'actions'])
            ->make(true);
    }

    /**
     * Store a newly created task.
     */
    public function store(Request $request, Project $project)
    {
        $validator = $this->validateTaskData($request);

        if ($validator->fails()) {
            return Error::response($validator->errors());
        }

        try {
            DB::beginTransaction();
            $task = $project->tasks()->create([
                'title' => $request->title,
                'description' => $request->description,
                'task_status_id' => $request->task_status_id,
                'task_priority_id' => $request->task_priority_id,
                'assigned_to_user_id' => $request->assigned_to_user_id,
                'due_date' => $request->due_date,
                'estimated_hours' => $request->estimated_hours,
                'is_milestone' => $request->boolean('is_milestone'),
                'parent_task_id' => $request->parent_task_id,
                'task_order' => $request->task_order ?? 0,
                'created_by_id' => auth()->id(),
            ]);

            DB::commit();

            return Success::response([
                'message' => __('Task created successfully!'),
                'task' => $task->load(['status', 'priority', 'assignedToUser']),
            ]);
        } catch (\Exception $e) {
            DB::rollBack();
            Log::error('Error creating project task: '.$e->getMessage());

            return Error::response(__('Failed to create task. Please try again.'));
        }
    }

    /**
     * Display the specified task.
     */
    public function show(Project $project, Task $task)
    {
        // Ensure task belongs to project using polymorphic relationship
        if ($task->taskable_type !== 'Modules\PMCore\app\Models\Project' || $task->taskable_id != $project->id) {
            return Error::response(__('Task not found.'));
        }

        $taskData = $task->load(['status', 'priority', 'assignedToUser']);

        // TODO: Need to fix the due date issue
        // Convert to array and properly format the due_date
        $taskArray = $taskData->toArray();
        if (! empty($taskArray['due_date'])) {
            // Parse the date and format it as Y-m-d only (no time component)
            $taskArray['due_date'] = \Carbon\Carbon::parse($taskArray['due_date'])->format('Y-m-d');
        }

        return Success::response([
            'task' => $taskArray,
        ]);
    }

    /**
     * Update the specified task.
     */
    public function update(Request $request, Project $project, Task $task)
    {
        // Ensure task belongs to project using polymorphic relationship
        if ($task->taskable_type !== 'Modules\PMCore\app\Models\Project' || $task->taskable_id != $project->id) {
            return Error::response(__('Task not found.'));
        }

        $validator = $this->validateTaskData($request);

        if ($validator->fails()) {
            return Error::response($validator->errors());
        }

        try {
            DB::beginTransaction();
            $task->update([
                'title' => $request->title,
                'description' => $request->description,
                'task_status_id' => $request->task_status_id,
                'task_priority_id' => $request->task_priority_id,
                'assigned_to_user_id' => $request->assigned_to_user_id,
                'due_date' => $request->due_date,
                'estimated_hours' => $request->estimated_hours,
                'is_milestone' => $request->boolean('is_milestone'),
                'parent_task_id' => $request->parent_task_id,
                'updated_by_id' => auth()->id(),
            ]);

            DB::commit();

            return Success::response([
                'message' => __('Task updated successfully!'),
                'task' => $task->load(['status', 'priority', 'assignedToUser']),
            ]);
        } catch (\Exception $e) {
            DB::rollBack();
            Log::error('Error updating project task: '.$e->getMessage());

            return Error::response(__('Failed to update task. Please try again.'));
        }
    }

    /**
     * Remove the specified task.
     */
    public function destroy(Project $project, Task $task)
    {
        // Ensure task belongs to project using polymorphic relationship
        if ($task->taskable_type !== 'Modules\PMCore\app\Models\Project' || $task->taskable_id != $project->id) {
            return Error::response(__('Task not found.'));
        }

        try {
            DB::beginTransaction();

            $task->delete();

            DB::commit();

            return Success::response([
                'message' => __('Task deleted successfully!'),
            ]);
        } catch (\Exception $e) {
            DB::rollBack();
            Log::error('Error deleting project task: '.$e->getMessage());

            return Error::response(__('Failed to delete task. Please try again.'));
        }
    }

    /**
     * Complete a task.
     */
    public function complete(Request $request, Project $project, Task $task)
    {
        // Ensure task belongs to project using polymorphic relationship
        if ($task->taskable_type !== 'Modules\PMCore\app\Models\Project' || $task->taskable_id != $project->id) {
            return Error::response(__('Task not found.'));
        }

        // Check if task is already completed
        if ($task->completed_at) {
            return Error::response(__('Task is already completed.'));
        }

        try {
            DB::beginTransaction();

            // Get completed status
            $completedStatus = TaskStatus::where('name', 'Completed')->first();
            if (! $completedStatus) {
                return Error::response(__('Completed status not found.'));
            }

            $task->update([
                'task_status_id' => $completedStatus->id,
                'completed_at' => now(),
                'completed_by' => auth()->id(),
                'updated_by_id' => auth()->id(),
            ]);

            DB::commit();

            return Success::response([
                'message' => __('Task completed successfully!'),
                'task' => $task->load(['status', 'priority', 'assignedToUser']),
            ]);
        } catch (\Exception $e) {
            DB::rollBack();
            Log::error('Error completing project task: '.$e->getMessage());

            return Error::response(__('Failed to complete task. Please try again.'));
        }
    }

    /**
     * Start a task.
     */
    public function start(Request $request, Project $project, Task $task)
    {
        // Ensure task belongs to project using polymorphic relationship
        if ($task->taskable_type !== 'Modules\PMCore\app\Models\Project' || $task->taskable_id != $project->id) {
            return Error::response(__('Task not found.'));
        }

        // Check if task is already completed
        if ($task->completed_at) {
            return Error::response(__('Cannot start a completed task.'));
        }

        // Check if task is already started
        if ($task->time_started_at) {
            return Error::response(__('Task is already started.'));
        }

        try {
            DB::beginTransaction();

            $task->update([
                'time_started_at' => now(),
                'updated_by_id' => auth()->id(),
            ]);

            DB::commit();

            return Success::response([
                'message' => __('Task started successfully!'),
                'task' => $task->load(['status', 'priority', 'assignedToUser']),
            ]);
        } catch (\Exception $e) {
            DB::rollBack();
            Log::error('Error starting project task: '.$e->getMessage());

            return Error::response(__('Failed to start task. Please try again.'));
        }
    }

    /**
     * Stop a task.
     */
    public function stop(Request $request, Project $project, Task $task)
    {
        // Ensure task belongs to project using polymorphic relationship
        if ($task->taskable_type !== 'Modules\PMCore\app\Models\Project' || $task->taskable_id != $project->id) {
            return Error::response(__('Task not found.'));
        }

        // Check if task is already completed
        if ($task->completed_at) {
            return Error::response(__('Cannot stop a completed task.'));
        }

        // Check if task is not started
        if (! $task->time_started_at) {
            return Error::response(__('Task is not currently running.'));
        }

        try {
            DB::beginTransaction();

            // Calculate actual hours if task was started
            $actualHours = null;
            if ($task->time_started_at) {
                $startTime = $task->time_started_at;
                $endTime = now();
                $actualHours = $endTime->diffInHours($startTime, true);
            }

            $task->update([
                'time_started_at' => null,
                'actual_hours' => $actualHours,
                'updated_by_id' => auth()->id(),
            ]);

            DB::commit();

            return Success::response([
                'message' => __('Task stopped successfully!'),
                'task' => $task->load(['status', 'priority', 'assignedToUser']),
            ]);
        } catch (\Exception $e) {
            DB::rollBack();
            Log::error('Error stopping project task: '.$e->getMessage());

            return Error::response(__('Failed to stop task. Please try again.'));
        }
    }

    /**
     * Reorder tasks in the board.
     */
    public function reorder(Request $request, Project $project)
    {
        // Handle single task reordering (from drag and drop)
        if ($request->has('task_id')) {
            $validator = Validator::make($request->all(), [
                'task_id' => 'required|exists:crm_tasks,id',
                'task_status_id' => 'required|exists:task_statuses,id',
                'order' => 'integer|min:0',
            ]);

            if ($validator->fails()) {
                return Error::response($validator->errors());
            }

            try {
                DB::beginTransaction();

                $task = Task::find($request->task_id);

                // Ensure task belongs to project using polymorphic relationship
                if ($task->taskable_type !== 'Modules\PMCore\app\Models\Project' || $task->taskable_id != $project->id) {
                    return Error::response(__('Task not found or does not belong to this project.'));
                }

                $task->update([
                    'task_status_id' => $request->task_status_id,
                    'task_order' => $request->order ?? 0,
                    'updated_by_id' => auth()->id(),
                ]);

                DB::commit();

                return Success::response([
                    'message' => __('Task status updated successfully!'),
                ]);
            } catch (\Exception $e) {
                DB::rollBack();
                Log::error('Error updating task status: '.$e->getMessage());

                return Error::response(__('Failed to update task status. Please try again.'));
            }
        }

        // Handle bulk task reordering
        $validator = Validator::make($request->all(), [
            'tasks' => 'required|array',
            'tasks.*.id' => 'required|exists:crm_tasks,id',
            'tasks.*.task_status_id' => 'required|exists:task_statuses,id',
            'tasks.*.task_order' => 'required|integer|min:0',
        ]);

        if ($validator->fails()) {
            return Error::response($validator->errors());
        }

        try {
            DB::beginTransaction();

            foreach ($request->tasks as $taskData) {
                $task = Task::find($taskData['id']);

                // Ensure task belongs to project using polymorphic relationship
                if ($task->taskable_type !== 'Modules\PMCore\app\Models\Project' || $task->taskable_id != $project->id) {
                    continue;
                }

                $task->update([
                    'task_status_id' => $taskData['task_status_id'],
                    'task_order' => $taskData['task_order'],
                    'updated_by_id' => auth()->id(),
                ]);
            }

            DB::commit();

            return Success::response([
                'message' => __('Tasks reordered successfully!'),
            ]);
        } catch (\Exception $e) {
            DB::rollBack();
            Log::error('Error reordering project tasks: '.$e->getMessage());

            return Error::response(__('Failed to reorder tasks. Please try again.'));
        }
    }

    /**
     * Validate task data.
     */
    private function validateTaskData(Request $request)
    {
        // Get project from route parameter
        $project = request()->route('project');
        $projectMemberUserIds = $project->members()->pluck('user_id')->toArray();

        $rules = [
            'title' => 'required|string|max:255',
            'description' => 'nullable|string',
            'task_status_id' => 'required|exists:task_statuses,id',
            'task_priority_id' => 'nullable|exists:task_priorities,id',
            'assigned_to_user_id' => 'nullable|exists:users,id',
            'due_date' => 'nullable|date',
            'estimated_hours' => 'nullable|numeric|min:0',
            'is_milestone' => 'boolean',
            'parent_task_id' => 'nullable|exists:crm_tasks,id',
            'task_order' => 'nullable|integer|min:0',
        ];

        // Add custom validation for project members
        if ($request->has('assigned_to_user_id') && $request->assigned_to_user_id) {
            $rules['assigned_to_user_id'] = [
                'exists:users,id',
                function ($attribute, $value, $fail) use ($projectMemberUserIds) {
                    if (! in_array($value, $projectMemberUserIds)) {
                        $fail('The selected user must be a member of this project.');
                    }
                },
            ];
        }

        return Validator::make($request->all(), $rules);
    }
}
