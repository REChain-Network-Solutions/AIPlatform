<?php

namespace Workdo\Taskly\Http\Controllers\Api;

use Illuminate\Contracts\Support\Renderable;
use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Illuminate\Support\Facades\DB;
use Workdo\Taskly\Entities\Stage;
use Workdo\Taskly\Entities\ClientProject;
use Workdo\Taskly\Entities\Task;
use Workdo\Taskly\Entities\Project;
use Workdo\Taskly\Entities\UserProject;
use Illuminate\Support\Facades\Auth;
use Workdo\Taskly\Entities\ActivityLog;
use Workdo\Taskly\Entities\BugComment;
use Workdo\Taskly\Entities\BugFile;
use Workdo\Taskly\Entities\BugReport;
use Workdo\Taskly\Entities\BugStage;
use Workdo\Taskly\Entities\Comment;
use Workdo\Taskly\Entities\Milestone;
use Workdo\Taskly\Entities\ProjectFile;
use Workdo\Taskly\Entities\SubTask;
use Workdo\Taskly\Entities\TaskFile;
use App\Models\Notification;
use App\Models\User;

class ProjectApiController extends Controller
{

    public function index(Request $request)
    {
        try {

            $validator = \Validator::make(
                $request->all(), [
                    'type'        => 'in:Ongoing,Finished,OnHold,All',
                    'workspace_id'  => 'required',
                ]
            );

            if($validator->fails())
            {
                $messages = $validator->getMessageBag();

                return response()->json(['status' => 0, 'message' => $messages->first()] , 403);
            }

            $objUser          = Auth::user();
            $currentWorkspace = $request->workspace_id;

            if (Auth::user()->hasRole('client')) {

                $projects = Project::select('projects.*')->join('client_projects', 'projects.id', '=', 'client_projects.project_id')
                                    ->projectonly()
                                    ->where('client_projects.client_id', '=', $objUser->id)
                                    ->where('projects.workspace', '=', $currentWorkspace);
            } else {

                $projects = Project::select('projects.*')->join('user_projects', 'projects.id', '=', 'user_projects.project_id')
                                    ->projectonly()
                                    ->where('user_projects.user_id', '=', $objUser->id)
                                    ->where('projects.workspace', '=', $currentWorkspace);

            }

            if ($request->has('type') && $request->type !== 'All') {

                $projects->where('status','=',$request->type);

            }

            $projects = $projects->get()->map(function($project){
                return [
                    'id'            => $project->id,
                    'name'          => $project->name,
                    'status'        => $project->status,
                    'description'   => $project->description,
					'total_task'    => $project->countTask(),
                	'total_comments'=> $project->countTaskComments(),
                    'start_date'    => $project->start_date,
                    'end_date'      => $project->end_date,
                    'created_by'    => $project->created_by,
					'members'           => $project->users->map(function($user){
                                            return  [
                                                'id'        => $user->id,
                                                'name'      => $user->name,
                                                'email'     => $user->email,
                                                'avatar'    => get_file($user->avatar),
                                            ];
                                        }),

                    'clients'           => $project->clients->map(function($client){
                                            return  [
                                                'id'        => $client->id,
                                                'name'      => $client->name,
                                                'email'     => $client->email,
                                                'avatar'    => get_file($client->avatar),
                                            ];
                                        }),
                ];

            });

            return response()->json([

                'status' => 1,
                'data'  => [
                    'projects'  => $projects,
                ]

            ]);

        } catch (\Exception $e) {
            return response()->json(['status'=>0,'message'=>'something went wrong!!!']);
        }
    }

    public function projectDetails(Request $request)
    {
        try{

            $validator = \Validator::make(
                $request->all(), [
                    'workspace_id'  => 'required',
                    'project_id'    => 'required',
                ]
            );

            if($validator->fails())
            {
                $messages = $validator->getMessageBag();

                return response()->json(['status'=>0, 'message'=>$messages->first()],403);
            }

            $objUser          = Auth::user();
            $currentWorkspace = $request->workspace_id;

            $project = Project::find($request->project_id);
            $daysleft = round((((strtotime($project->end_date) - strtotime(date('Y-m-d'))) / 24) / 60) / 60);
            $totalMembers = (int) $project->users->count() + (int) $project->clients->count() + (int) $project->venders->count();


            $projectDetails = [
                'id'                => $project->id,
                'name'              => $project->name,
                'start_date'        => $project->start_date,
                'end_date'          => $project->end_date,
                'status'            => $project->status,
                'total_members'     => $totalMembers,
                'description'       => $project->description,
                'daysleft'          => $daysleft,
                'budget'            => number_format($project->budget),
                'total_task'        => $project->countTask(),
                'total_comments'    => $project->countTaskComments(),
                'project_copylink'  => route('project.shared.link', [\Illuminate\Support\Facades\Crypt::encrypt($project->id)]),

                'members'           => $project->users->map(function($user){
                                                            return  [
                                                                'id'        => $user->id,
                                                                'name'      => $user->name,
                                                                'email'     => $user->email,
                                                                'avatar'    => get_file($user->avatar),
                                                            ];
                                                        }),

                'clients'           => $project->clients->map(function($client){
                                                            return  [
                                                                'id'        => $client->id,
                                                                'name'      => $client->name,
                                                                'email'     => $client->email,
                                                                'avatar'    => get_file($client->avatar),
                                                            ];
                                                        }),

                'vendors'           => $project->venders->map(function($vender){
                                                            return  [
                                                                'id'        => $vender->id,
                                                                'name'      => $vender->name,
                                                                'email'     => $vender->email,
                                                                'avatar'    => get_file($vender->avatar),
                                                            ];
                                                        }),

                'milestones' => $project->milestones->map(function($milestone){
                                                            return  [
                                                                'id'            => $milestone->id,
                                                                'project_id'    => $milestone->project_id,
                                                                'title'         => $milestone->title,
                                                                'status'        => $milestone->status,
                                                                'cost'          => $milestone->cost,
                                                                'summary'       => $milestone->summary,
                                                                'progress'      => $milestone->progress,
                                                                'start_date'    => $milestone->start_date,
                                                                'end_date'      => $milestone->end_date,
                                                            ];
                                                        }),
            ];

            return response()->json([
                'status' => 1,
                'data'  => $projectDetails,

            ]);

        } catch (\Exception $e) {
            return response()->json(['status'=>0,'message'=>'something went wrong!!!']);
        }
    }

    public function getWorkspaceUsers(Request $request)
    {
        try {

            $validator = \Validator::make(
                $request->all(), [
                    'workspace_id'  => 'required',
                ]
            );

            if($validator->fails())
            {
                $messages = $validator->getMessageBag();

                return response()->json(['status' => 0 , 'message' => $messages->first()], 403);
            }

            $objUser          = Auth::user();
            $currentWorkspace = $request->workspace_id;

            $users  = User::where('created_by', creatorId())
                                    ->emp()
                                    ->where('workspace_id', $currentWorkspace)
                                    ->orWhere('id', Auth::user()->id)
                                    // ->limit(10)->offset((($request->page??1)-1)*10)
                                    ->get()
                                    ->map(function($user){
                                            return [
                                                'id' => $user->id,
                                                'name' => $user->name,
                                                'email' => $user->email,
                                            ];
                                        });

            return response()->json([
                'status' => 1,
                'data'   => $users,

            ]);

        } catch (\Exception $e) {
            return response()->json(['status' => 0 ,'message' => 'something went wrong!!!']);
        }
    }

    public function projectActivity(Request $request)
    {
        try {

            $validator = \Validator::make(
                $request->all(), [
                    'workspace_id'  => 'required',
                    'project_id'  => 'required',
                ]
            );

            if($validator->fails())
            {
                $messages = $validator->getMessageBag();

                return response()->json(['status'=> 0, 'message' => $messages->first()], 403);
            }

            $objUser          = Auth::user();
            $currentWorkspace = $request->workspace_id;

            $project    = Project::find($request->project_id);
            $activities = $project->activities->map(function($activity){
                return [
                        'id'        => $activity->id,
                        // 'log_type'  => $activity->log_type,
                        'remark'    => strip_tags($activity->getRemark()),
                        'time'      => $activity->created_at->diffForHumans(),
                ];
            });

            return response()->json([
                'status' => 1,
                'data'   => $activities,
            ]);

        } catch (\Exception $e) {
            return response()->json(['status' => 0, 'message' => 'something went wrong!!!']);
        }

    }

    public function projectCreateAndUpdate(Request $request)
    {

        try{

            if($request->project_id){

                $validator = \Validator::make(
                    $request->all(), [
                        'workspace_id'  => 'required',
                        'name'          => 'required',
                        'description'   => 'required',
                        'budget'        => 'gt:0|numeric',
                        'start_date'    => 'date_format:Y-m-d',
                        'end_date'      => 'date_format:Y-m-d',
                        'status'        => 'in:Ongoing,Finished,OnHold',
                    ]
                );

                if($validator->fails())
                {
                    $messages = $validator->getMessageBag();
                    return response()->json(['status' => 0, 'message' => $messages->first()], 403);
                }
                $objUser          = Auth::user();
                $currentWorkspace = $request->workspace_id;
                $post       = $request->all();

				$userList           = [];
                if (isset($post['users_list'])) {
                         $userList = $post['users_list'];
                    //$userList = explode(',', $post['users_list']);
					//$userList = json_decode($post['users_list'], true);
                }

                if (!empty($userList)) {
                    $user_project            = UserProject::where('project_id', '=', $request->project_id)->delete();
                }

                $project    = Project::find($request->project_id);
                $project->update($post);

				foreach ($userList as $email) {
                        $permission    = 'Member';
                        $registerUsers = User::where('active_workspace', $currentWorkspace)->where('email', $email)->first();
                        if ($registerUsers) {
                            if ($registerUsers->id == $objUser->id) {
                                $permission = 'Owner';
                            }
                            $this->inviteUser($registerUsers, $project, $permission);
                        }
                    }

                return response()->json(['status' => 1 , 'message' => 'Project Updated Successfully.'] , 200);

            }else{

                    $validator = \Validator::make(
                        $request->all(), [
                            'workspace_id'  => 'required',
                            'name'          => 'required',
                            'description'   => 'required',
                            'budget'        => 'gt:0|numeric',
                            'start_date'    => 'date_format:Y-m-d',
                            'end_date'      => 'date_format:Y-m-d',
                            'users_list'    => 'required'
                        ]
                    );

                    if($validator->fails())
                    {
                        $messages = $validator->getMessageBag();

                        return response()->json(['status'=>0, 'message'=>$messages->first()],403);
                    }

                    $objUser          = Auth::user();
                    $currentWorkspace = $request->workspace_id;

                    $post = $request->all();

                    $post['start_date']         = isset($request->start_date) ? $request->start_date : date('Y-m-d');
                    $post['end_date']           = isset($request->end_date)   ? $request->end_date   : date('Y-m-d');
                    $post['workspace']          = $currentWorkspace;
                    $post['created_by']         = $objUser->id;
                    $post['copylinksetting']    = '{"member":"on","client":"on","milestone":"off","progress":"off","basic_details":"on","activity":"off","attachment":"on","bug_report":"on","task":"off","invoice":"off","timesheet":"off" ,"password_protected":"off"}';

                     $userList           = [];
                    if (isset($post['users_list'])) {
                             $userList = $post['users_list'];
                        //$userList = explode(',', $post['users_list']);
						//$userList = json_decode($post['users_list'], true);
                    }

                    // $userList = [];
                    $userList[] = $objUser->email;
                    $userList   = array_unique($userList);
                    $objProject = Project::create($post);

                    foreach ($userList as $email) {
                        $permission    = 'Member';
                        $registerUsers = User::where('active_workspace', $currentWorkspace)->where('email', $email)->first();
                        if ($registerUsers) {
                            if ($registerUsers->id == $objUser->id) {
                                $permission = 'Owner';
                            }
                            $this->inviteUser($registerUsers, $objProject, $permission);
                        }
                    }

                return response()->json(['status' => 1,'message'=>'Project Created Successfully!'],200);
            }

        } catch (\Exception $e) {
            return response()->json(['status'=> 0 ,'message' => 'something went wrong!!!']);
        }

    }

    public function inviteUser($user, $project, $permission)
    {
        $arrData               = [];
        $arrData['user_id']    = $user->id;
        $arrData['project_id'] = $project->id;
        //$is_invited            = UserProject::where($arrData)->first();
        //if (!$is_invited) {
            UserProject::create($arrData);
        //}
    }

    public function projectStatusUpdate(Request $request)
    {
        try{

            $validator = \Validator::make(
                $request->all(), [
                    'workspace_id'  => 'required',
                    'project_id'    => 'required',
                    'status'        => 'required|in:Ongoing,Finished,OnHold',
                ]
            );

            if($validator->fails())
            {
                $messages = $validator->getMessageBag();

                return response()->json(['status'=> 0, 'message'=>$messages->first()],403);
            }

            $project            = Project::find($request->project_id);
            $project->status    = $request->status;
            $project->save();

            return response()->json(['status' => 1,'message'=>'Project Status Change Successfully!'],200);

        } catch (\Exception $e) {
            return response()->json(['status'=> 0 ,'message' => 'something went wrong!!!']);
        }
    }

    public function destroyProject(Request $request)
    {
        try{

            $objUser    = Auth::user();
            $projectID  = $request->project_id;

			$project = Project::where('workspace',$request->workspace_id)->where('id',$projectID);
			if($objUser->type != 'company'){
				$project = $project->where('created_by',$objUser->id);
			}

			$project = $project->first();

            if(!$project){
                return response()->json(['status'=>'error','message'=>'Project Not Found'],404);
            }

            if($project->created_by == $objUser->id || $objUser->type == 'company')
            {
                $task   = Task::where('project_id', '=', $project->id)->count();
                $bug    = BugReport::where('project_id', '=', $project->id)->count();

                if($task == 0 && $bug == 0)
                {
                    UserProject::where('project_id', '=', $projectID)->delete();
                    $ProjectFiles=ProjectFile::where('project_id', '=', $projectID)->get();
                    foreach($ProjectFiles as $ProjectFile){

                        delete_file($ProjectFile->file_path);
                        $ProjectFile->delete();
                    }

                    Milestone::where('project_id', '=', $projectID)->delete();
                    ActivityLog::where('project_id', '=', $projectID)->delete();

                    $project->delete();

                    return response()->json(['status'=> 1, 'message'=>'Project Deleted Successfully!']);
                }
                else
                {
                    return response()->json(['status'=> 0, 'message'=>'There are some Task and Bug on Project, please remove it first!']);
                }
            }
            else
            {
                return response()->json(['status'=> 0, 'message'=>"You can't Delete Project!"]);
            }

        } catch (\Exception $e) {
            return response()->json(['status'=> 0 ,'message' => 'something went wrong!!!']);
        }
    }
}
