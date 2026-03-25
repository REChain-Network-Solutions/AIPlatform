<?php

namespace Workdo\Taskly\Http\Controllers\Api;

use Illuminate\Contracts\Support\Renderable;
use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Illuminate\Support\Facades\DB;
use Workdo\Taskly\Entities\ClientProject;
use Workdo\Taskly\Entities\Stage;
use Workdo\Taskly\Entities\Task;
use Workdo\Taskly\Entities\UserProject;
use Illuminate\Support\Facades\Auth;
use App\Models\Notification;

class ProjectDashboardApiController extends Controller
{
    public function __construct()
    {

    }

    public function index(Request $request)
    {
        try
        {

            $userObj          = Auth::user();
            $currentWorkspace = $request->workspace_id;

            if(Auth::user()->hasRole('client'))
            {
                $doneStage    = Stage::where('workspace_id', '=', $currentWorkspace)
                                        ->where('complete', '=', '1')
                                        ->first();

                $totalProject   = ClientProject::join("projects", "projects.id", "=", "client_projects.project_id")
                                                ->where("client_id", "=", $userObj->id)
                                                ->where('projects.workspace', '=', $currentWorkspace)
                                                ->where('projects.type', 'project')
                                                ->count();

                $totalBugs      = ClientProject::join("bug_reports", "bug_reports.project_id", "=", "client_projects.project_id")
                                                ->join("projects", "projects.id", "=", "client_projects.project_id")
                                                ->where('projects.workspace', '=', $currentWorkspace)
                                                ->where('projects.type', 'project')
                                                ->count();

                $totalTask      = ClientProject::join("tasks", "tasks.project_id", "=", "client_projects.project_id")
                                                ->join("projects", "projects.id", "=", "client_projects.project_id")
                                                ->where('projects.workspace', '=', $currentWorkspace)
                                                ->where('projects.type', 'project')
                                                ->where("client_id", "=", $userObj->id)
                                                ->count();

                if(!empty($doneStage))
                {
                    $completeTask   = ClientProject::join("tasks", "tasks.project_id", "=", "client_projects.project_id")
                                                    ->join("projects", "projects.id", "=", "client_projects.project_id")
                                                    ->where('projects.workspace', '=', $currentWorkspace)
                                                    ->where('projects.type', 'project')
                                                    ->where("client_id", "=", $userObj->id)
                                                    ->where('tasks.status', '=', $doneStage->id)
                                                    ->count();
                }
                else
                {
                    $completeTask = 0;
                }

                    $tasks          = Task::select(['tasks.*','stages.name as status','stages.complete',])
                                            ->join("client_projects", "tasks.project_id", "=", "client_projects.project_id")
                                            ->join("projects", "projects.id", "=", "client_projects.project_id")
                                            ->join("stages", "stages.id", "=", "tasks.status")
                                            ->where('projects.workspace', '=', $currentWorkspace)
                                            ->where("client_id", "=", $userObj->id)
                                            ->orderBy('tasks.id', 'desc')
                                            ->where('projects.type', 'project')
                                            ->limit(5)
                                            ->with('project')
                                            ->get()
                                            ->map(function($task){
                                                return [
                                                    "id"            => $task->id,
                                                    "title"         => $task->title,
                                                    "priority"      => $task->priority,
                                                    "start_date"    => $task->start_date,
                                                    "due_date"      => $task->due_date,
                                                    "project_name"  => $task->project->name,
                                                    "project_id"  => $task->project->id,
                                                    "status"        => $task->status,
                                                ];
                                            });

                    $totalMembers   = 0 ;

                    $projectProcess = ClientProject::join("projects", "projects.id", "=", "client_projects.project_id")
                                                    ->where('projects.workspace', '=', $currentWorkspace)
                                                    ->where('projects.type', 'project')
                                                    ->where("client_id", "=", $userObj->id)
                                                    ->groupBy('projects.status')
                                                    ->selectRaw('count(projects.id) as count, projects.status')
                                                    ->pluck('count', 'projects.status');

                    $status   = [];
                    if(count($projectProcess) > 0)
                    {
                        foreach($projectProcess as $lable => $process)
                        {
                            if($totalProject == 0)
                            {
                                $status[$lable] = 0.00;
                            }
                            else
                            {
                                $status[$lable] = round(($process * 100) / $totalProject, 2);
                            }
                        }
                    }
                    else
                    {
                            $status[$lable]   = 100;
                    }

                    return response()->json([

                        'status' => 1,
                        'data'  => [
                            'totalProject'  => $totalProject,
                            'totalBugs'     => $totalBugs,
                            'totalTask'     => $totalTask,
                            'totalMembers'  => $totalMembers,
                            'status'        => $status,
                            'tasks'         => $tasks,
                        ]

                    ]);

            }

            $totalProject = UserProject::join("projects", "projects.id", "=", "user_projects.project_id")
                                        ->where("user_id", "=", $userObj->id)
                                        ->where('projects.workspace', '=', $currentWorkspace)
                                        ->where('projects.type', 'project')
                                        ->count();

            $doneStage    = Stage::where('workspace_id', '=', $currentWorkspace)
                                    ->where('complete', '=', '1')
                                    ->first();

            $totalBugs    = UserProject::join("bug_reports", "bug_reports.project_id", "=", "user_projects.project_id")
                                        ->join("projects", "projects.id", "=", "user_projects.project_id")
                                        ->where("user_id", "=", $userObj->id)
                                        ->where('projects.type', 'project')
                                        ->where('projects.workspace', '=', $currentWorkspace)
                                        ->count();

            $totalTask    = UserProject::join("tasks", "tasks.project_id", "=", "user_projects.project_id")
                                        ->join("projects", "projects.id", "=", "user_projects.project_id")
                                        ->where("user_id", "=", $userObj->id)
                                        ->where('projects.workspace', '=', $currentWorkspace)
                                        ->where('projects.type', 'project')
                                        ->count();

            $totalMembers = UserProject::join('projects', 'projects.id', '=', 'user_projects.project_id')
                                        ->where('projects.workspace','=', $currentWorkspace)
                                        ->count();

            if(!empty($doneStage))
            {
                $completeTask = UserProject::join("tasks", "tasks.project_id", "=", "user_projects.project_id")
                                            ->join("projects", "projects.id", "=", "user_projects.project_id")
                                            ->where("user_id", "=", $userObj->id)
                                            ->where('projects.workspace', '=', $currentWorkspace)
                                            ->where('tasks.status', '=', $doneStage->id)
                                            ->where('projects.type', 'project')
                                            ->count();
            }
            else
            {
                $completeTask = 0;
            }

            $tasks        = Task::select(['tasks.*','stages.name as status','stages.complete',])
                                    ->join("user_projects", "tasks.project_id", "=", "user_projects.project_id")
                                    ->join("projects", "projects.id", "=", "user_projects.project_id")
                                    ->join("stages", "stages.id", "=", "tasks.status")
                                    ->where("user_id", "=", $userObj->id)
                                    ->where('projects.workspace', '=', $currentWorkspace)
                                    ->orderBy('tasks.id', 'desc')
                                    ->where('projects.type', 'project')
                                    ->limit(5)
                                    ->with('project')
                                    ->get()
                                    ->map(function($task){
                                        return [
                                            "id"            => $task->id,
                                            "title"         => $task->title,
                                            "priority"      => $task->priority,
                                            "start_date"    => $task->start_date,
                                            "due_date"      => $task->due_date,
                                            "project_name"  => $task->project->name,
                                            "project_id"    => $task->project->id,
                                            "status"        => $task->status,
                                        ];
                                    });

            $projectProcess  = UserProject::join("projects", "projects.id", "=", "user_projects.project_id")
                                            ->where("user_id", "=", $userObj->id)
                                            ->where('projects.workspace', '=', $currentWorkspace)
                                            ->where('projects.type', 'project')
                                            ->groupBy('projects.status')
                                            ->selectRaw('count(projects.id) as count, projects.status')
                                            ->pluck('count', 'projects.status');


                $status   = [];

                if(count($projectProcess) > 0)
                {
                    foreach($projectProcess as $lable => $process)
                    {
                        if($totalProject == 0)
                        {
                            $status[$lable] = 0.00;
                        }
                        else
                        {
                            $status[$lable] = (int) (($process * 100) / $totalProject);
                        }
                    }

                }
                else
                {

                    $status['Ongoing']   = 100;
                }

                $data = [
                    'totalProject' => $totalProject,
                    'totalBugs' => $totalBugs,
                    'totalTask' => $totalTask,
                    'totalMembers' => $totalMembers,
                    'status' => $status,
                    'tasks' => $tasks,
                ];

                return response()->json(['status' => 1,'data'  => $data]);

        } catch (\Exception $e) {
            return response()->json(['status'=>0,'message'=>'something went wrong!!!']);
        }
    }


}
