<?php

use App\Models\Role;
use App\Models\User;
use App\Models\Module;
use App\Models\Company;
use App\Models\Permission;
use App\Models\ModuleSetting;
use App\Models\PermissionRole;
use App\Models\PermissionType;
use App\Models\UserPermission;
use Modules\Feedback\Entities\Feedback;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{

    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        // create module and permissions
        $permissions = [
            [
                'name' => 'add_feedback',
                'display_name' => 'Add Feedback'
            ],
            [
                'name' => 'view_feedback',
                'display_name' => 'View Feedback'
            ],
            [
                'name' => 'edit_feedback',
                'display_name' => 'Edit Feedback'
            ],
            [
                'name' => 'delete_feedback',
                'display_name' => 'Delete Feedback'
            ]
        ];

        $module = Module::where('module_name', 'feedback')->first();
        if (!$module) {
            $module = new Module();
            $module->module_name = 'feedback';
            $module->description = 'User can view all Feedback.';
            $module->saveQuietly();
        }

        foreach ($permissions as $index => $permission) {
            $data_permission = Permission::where('name', $permission['name'])->first();
            if ($data_permission) {
                unset($permissions[$index]);
            }
        }

        if (count($permissions) > 0) {
            $module->permissions()->createMany($permissions);
        }

        $all = ['add_feedback', 'view_feedback', 'edit_feedback', 'delete_feedback'];
        Permission::whereIn('name', $all)->update(['allowed_permissions' => Permission::ALL_4_OWNED_2_NONE_5]);

        $companies = Company::all();

        // We will insert these for the new company from event listener
        foreach ($companies as $company) {
            // Feedback::addModuleSetting($company);
            $roles = ['client', 'employee', 'admin'];
            ModuleSetting::createRoleSettingEntry('feedback', $roles, $company);
        }

        Artisan::call('module:enable feedback');
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        $module = Module::where('module_name', 'feedback')->first();
        $permisisons = Permission::where('module_id', $module->id)->get()->pluck('id')->toArray();
        PermissionRole::whereIn('permission_id', $permisisons)->delete();

        Module::where('module_name', 'feedback')->delete();
        ModuleSetting::where('module_name', 'feedback')->delete();
    }
};
