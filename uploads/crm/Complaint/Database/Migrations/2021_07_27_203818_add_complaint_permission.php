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
use Modules\Complaint\Entities\Complaint;
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
                'name' => 'add_complaint',
                'display_name' => 'Add Complaint'
            ],
            [
                'name' => 'view_complaint',
                'display_name' => 'View Complaint'
            ],
            [
                'name' => 'edit_complaint',
                'display_name' => 'Edit Complaint'
            ],
            [
                'name' => 'delete_complaint',
                'display_name' => 'Delete Complaint'
            ]
        ];

        $module = Module::where('module_name', 'complaint')->first();
        if (!$module) {
            $module = new Module();
            $module->module_name = 'complaint';
            $module->description = 'User can view all Complaint.';
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

        $all = ['add_complaint', 'view_complaint', 'edit_complaint', 'delete_complaint'];
        Permission::whereIn('name', $all)->update(['allowed_permissions' => Permission::ALL_4_OWNED_2_NONE_5]);

        $companies = Company::all();

        // We will insert these for the new company from event listener
        foreach ($companies as $company) {
            // Complaint::addModuleSetting($company);
            $roles = ['client', 'employee', 'admin'];
            ModuleSetting::createRoleSettingEntry('complaint', $roles, $company);
        }

        Artisan::call('module:enable complaint');
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        $module = Module::where('module_name', 'complaint')->first();
        $permisisons = Permission::where('module_id', $module->id)->get()->pluck('id')->toArray();
        PermissionRole::whereIn('permission_id', $permisisons)->delete();

        Module::where('module_name', 'complaint')->delete();
        ModuleSetting::where('module_name', 'complaint')->delete();
    }
};
