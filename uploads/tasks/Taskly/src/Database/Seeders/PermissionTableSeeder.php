<?php

namespace Workdo\Taskly\Database\Seeders;

use App\Models\Role;
use App\Models\Permission;
use Illuminate\Database\Seeder;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\Artisan;

class PermissionTableSeeder extends Seeder
{
     public function run()
    {
        Model::unguard();
        Artisan::call('cache:clear');
        $module = 'Taskly';

        $permissions = [
            'taskly manage',
            'taskly setup manage',
            'taskly dashboard manage',
            'project manage',
            'project create',
            'project edit',
            'project delete',
            'project show',
            'project invite user',
            'project report manage',
            'project import',
            'project setting',
            'project finance manage',
            'team member remove',
            'team client remove',
            'bug manage',
            'bug create',
            'bug edit',
            'bug delete',
            'bug show',
            'bug move',
            'bug comments create',
            'bug comments delete',
            'bug file uploads',
            'bug file delete',
            'bugstage manage',
            'bugstage edit',
            'bugstage delete',
            'bugstage show',
            'milestone manage',
            'milestone create',
            'milestone edit',
            'milestone delete',
            'milestone show',
            'task manage',
            'task create',
            'task edit',
            'task delete',
            'task show',
            'task move',
            'task file manage',
            'task file uploads',
            'task file delete',
            'task file show',
            'task comment manage',
            'task comment create',
            'task comment edit',
            'task comment delete',
            'task comment show',
            'taskstage manage',
            'taskstage edit',
            'taskstage delete',
            'taskstage show',
            'sub-task manage',
            'sub-task create',
            'sub-task edit',
            'sub-task delete',

        ];

        $company_role = Role::where('name','company')->first();
        foreach ($permissions as $key => $value)
        {
            $check = Permission::where('name',$value)->where('module',$module)->exists();
            if($check == false)
            {
                $permission = Permission::create(
                    [
                        'name' => $value,
                        'guard_name' => 'web',
                        'module' => $module,
                        'created_by' => 0,
                        "created_at" => date('Y-m-d H:i:s'),
                        "updated_at" => date('Y-m-d H:i:s')
                    ]
                );
                if(!$company_role->hasPermission($value))
                {
                    $company_role->givePermission($permission);
                }
            }
        }
    }
}
