<?php

namespace Modules\PMCore\app\Policies;

use App\Models\Task;
use App\Models\User;
use Illuminate\Auth\Access\HandlesAuthorization;
use Modules\PMCore\app\Models\Project;

class ProjectTaskPolicy
{
    use HandlesAuthorization;

    /**
     * Determine whether the user can view any tasks.
     */
    public function viewAny(User $user, Project $project): bool
    {
        // Can view tasks if can view the project
        return $user->can('pmcore.view-project-tasks') &&
               app(ProjectPolicy::class)->view($user, $project);
    }

    /**
     * Determine whether the user can view the task.
     */
    public function view(User $user, Task $task): bool
    {
        if (! $user->can('pmcore.view-project-tasks')) {
            return false;
        }

        $project = Project::find($task->model_id);

        return $project && app(ProjectPolicy::class)->view($user, $project);
    }

    /**
     * Determine whether the user can create tasks.
     */
    public function create(User $user, Project $project): bool
    {
        return $user->can('pmcore.create-project-task') &&
               app(ProjectPolicy::class)->view($user, $project);
    }

    /**
     * Determine whether the user can update the task.
     */
    public function update(User $user, Task $task): bool
    {
        $project = Project::find($task->model_id);
        if (! $project) {
            return false;
        }

        if ($user->can('pmcore.edit-project-task')) {
            return app(ProjectPolicy::class)->view($user, $project);
        }

        if ($user->can('pmcore.edit-own-task')) {
            return $task->assigned_to === $user->id;
        }

        return false;
    }

    /**
     * Determine whether the user can delete the task.
     */
    public function delete(User $user, Task $task): bool
    {
        $project = Project::find($task->model_id);

        return $project &&
               $user->can('pmcore.delete-project-task') &&
               app(ProjectPolicy::class)->view($user, $project);
    }

    /**
     * Determine whether the user can assign the task.
     */
    public function assign(User $user, Task $task): bool
    {
        $project = Project::find($task->model_id);

        return $project &&
               $user->can('pmcore.assign-project-task') &&
               app(ProjectPolicy::class)->view($user, $project);
    }

    /**
     * Determine whether the user can complete the task.
     */
    public function complete(User $user, Task $task): bool
    {
        return $user->can('pmcore.complete-project-task') &&
               ($task->assigned_to === $user->id || $user->can('pmcore.edit-project-task'));
    }

    /**
     * Determine whether the user can start the task.
     */
    public function start(User $user, Task $task): bool
    {
        return $user->can('pmcore.start-project-task') &&
               ($task->assigned_to === $user->id || $user->can('pmcore.edit-project-task'));
    }

    /**
     * Determine whether the user can reorder tasks.
     */
    public function reorder(User $user, Project $project): bool
    {
        return $user->can('pmcore.reorder-project-tasks') &&
               app(ProjectPolicy::class)->view($user, $project);
    }
}
