<?php

namespace Workdo\Taskly\Entities;

use App\Models\Permission;
use App\Models\Role;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use App\Models\User;
use App\Models\WorkSpace;


class TaskUtility extends Model
{
    use HasFactory;

    public static function GivePermissionToRoles($role_id = null,$rolename = null)
    {
        $client_permissions=[
            'taskly dashboard manage',
            'taskly manage',
            'project manage',
            'project create',
            'project show',
            'bug edit',
            'bug show',
            'bug move',
            'bug manage',
            'bug create',
            'bug delete',
            'bug comments delete',
            'bug comments create',
            'bug file delete',
            'bug file uploads',
            'milestone manage',
            'milestone show',
            'sub-task delete',
            'sub-task edit',
            'sub-task manage',
            'task file uploads',
            'task file manage',
            'task file show',
            'task file delete',
            'task comment show',
            'task comment delete',
            'task comment create',
            'task comment edit',
            'task comment manage',
            'task show',
            'task delete',
            'task edit',
            'task create',
            'task manage',
        ];


        $staff_permissions=[
            'taskly dashboard manage',
            'taskly manage',
            'project manage',
            'project show',
            'bug show',
            'bug move',
            'bug manage',
            'bug comments create',
            'bug file delete',
            'bug file uploads',
            'sub-task edit',
            'sub-task manage',
            'task file manage',
            'task file show',
            'task comment show',
            'task comment manage',
            'task show',
            'task manage',
        ];

        if($role_id == Null)
        {
            // client
            $roles_c = Role::where('name','client')->get();
            foreach($roles_c as $role)
            {
                foreach($client_permissions as $permission_c){
                    $permission = Permission::where('name',$permission_c)->first();
                    if(!$role->hasPermission($permission_c))
                    {
                        $role->givePermission($permission);
                    }
                }
            }

            // vender
            $roles_s = Role::where('name','staff')->get();

            foreach($roles_s as $role)
            {
                foreach($staff_permissions as $permission_s){
                    $permission = Permission::where('name',$permission_s)->first();
                    if(!$role->hasPermission($permission_s))
                    {
                        $role->givePermission($permission);
                    }
                }
            }

        }
        else
        {
            if($rolename == 'client')
            {
                $roles_c = Role::where('name','client')->where('id',$role_id)->first();
                foreach($client_permissions as $permission_c){
                    $permission = Permission::where('name',$permission_c)->first();
                    if(!$roles_c->hasPermission($permission_c))
                    {
                        $roles_c->givePermission($permission);
                    }
                }
            }
            elseif($rolename == 'staff')
            {
                $roles_s = Role::where('name','staff')->where('id',$role_id)->first();
                foreach($staff_permissions as $permission_s){
                    $permission = Permission::where('name',$permission_s)->first();
                    if(!$roles_s->hasPermission($permission_s))
                    {
                        $roles_s->givePermission($permission);
                    }
                }
            }
        }

    }

    public static function defaultdata($company_id = null,$workspace_id = null)
    {
        $bug_stages = [
            '#77b6ea' => 'Unconfirmed',
            '#6e00ff' => 'Confirmed',
            '#3cb8d9' => 'In Progress',
            '#37b37e' => 'Resolved',
            '#545454' => 'Verified',
        ];
        $task_stages = [
            '#77b6ea' => 'Todo',
            '#545454' => 'In Progress',
            '#3cb8d9' => 'Review',
            '#37b37e' => 'Done',
        ];
        $key = 0;
        $key1 = 0;
        $bug_lastKey       = count($bug_stages) - 1;
        $task_lastKey       = count($task_stages) - 1;

        if($company_id == Null)
        {
            $companys = User::where('type','company')->get();
            foreach($companys as $company)
            {
                $WorkSpaces = WorkSpace::where('created_by',$company->id)->get();
                foreach($WorkSpaces as $WorkSpace)
                {
                    foreach($task_stages as $color => $stage)
                    {
                        $taskstage = Stage::where('name',$stage)->where('workspace_id',$WorkSpace->id)->where('created_by',$company->id)->first();
                        if($taskstage == null){
                            Stage::create([
                                    'name' => $stage,
                                    'color' => $color,
                                    'workspace_id' => !empty($WorkSpace->id) ? $WorkSpace->id : 0,
                                    'created_by' => !empty($company->id) ? $company->id : 2,
                                    'complete' => ($key == $task_lastKey) ? true : false,
                                    'order' => $key1,
                                ]);
                            $key1++;
                        }
                    }
                    foreach($bug_stages as $color => $stage)
                    {
                        $bugstage = BugStage::where('name',$stage)->where('workspace_id',$WorkSpace->id)->where('created_by',$company->id)->first();

                        if($bugstage == null){
                            BugStage::create([
                                'name' => $stage,
                                'color' => $color,
                                'workspace_id' =>   !empty($WorkSpace->id) ? $WorkSpace->id : 0,
                                'created_by' => !empty($company->id) ? $company->id : 2,
                                'complete' => ($key == $bug_lastKey) ? true : false,
                                'order' => $key,
                            ]);
                        $key++;
                        }
                    }
                }
            }
        }elseif($workspace_id == Null){
            $company = User::where('type','company')->where('id',$company_id)->first();
            $WorkSpaces = WorkSpace::where('created_by',$company->id)->get();
            foreach($WorkSpaces as $WorkSpace)
            {
                foreach($task_stages as $color => $stage)
                {
                    $taskstage = Stage::where('name',$stage)->where('workspace_id',$WorkSpace->id)->where('created_by',$company->id)->first();
                    if($taskstage == null){
                        Stage::create([
                                'name' => $stage,
                                'color' => $color,
                                'workspace_id' => !empty($WorkSpace->id) ? $WorkSpace->id : 0,
                                'created_by' => !empty($company->id) ? $company->id : 2,
                                'complete' => ($key == $task_lastKey) ? true : false,
                                'order' => $key1,
                            ]);
                        $key1++;
                    }
                }
                foreach($bug_stages as $color => $stage)
                {
                    $bugstage = BugStage::where('name',$stage)->where('workspace_id',$WorkSpace->id)->where('created_by',$company->id)->first();

                    if($bugstage == null){
                        BugStage::create([
                            'name' => $stage,
                            'color' => $color,
                            'workspace_id' =>   !empty($WorkSpace->id) ? $WorkSpace->id : 0,
                            'created_by' => !empty($company->id) ? $company->id : 2,
                            'complete' => ($key == $bug_lastKey) ? true : false,
                            'order' => $key,
                        ]);
                    $key++;
                    }
                }
            }
        }else{
            $company = User::where('type','company')->where('id',$company_id)->first();
            $WorkSpace = WorkSpace::where('created_by',$company->id)->where('id',$workspace_id)->first();
            foreach($task_stages as $color => $stage)
            {
                $taskstage = Stage::where('name',$stage)->where('workspace_id',$WorkSpace->id)->where('created_by',$company->id)->first();
                if($taskstage == null){
                    Stage::create([
                            'name' => $stage,
                            'color' => $color,
                            'workspace_id' => !empty($WorkSpace->id) ? $WorkSpace->id : 0,
                            'created_by' => !empty($company->id) ? $company->id : 2,
                            'complete' => ($key == $task_lastKey) ? true : false,
                            'order' => $key1,
                        ]);
                    $key1++;
                }
            }
            foreach($bug_stages as $color => $stage)
            {
                $bugstage = BugStage::where('name',$stage)->where('workspace_id',$WorkSpace->id)->where('created_by',$company->id)->first();

                if($bugstage == null){
                    BugStage::create([
                        'name' => $stage,
                        'color' => $color,
                        'workspace_id' =>   !empty($WorkSpace->id) ? $WorkSpace->id : 0,
                        'created_by' => !empty($company->id) ? $company->id : 2,
                        'complete' => ($key == $bug_lastKey) ? true : false,
                        'order' => $key,
                    ]);
                $key++;
                }
            }
        }
    }
}
