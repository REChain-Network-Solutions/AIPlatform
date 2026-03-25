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
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;
use Modules\Engineerings\Entities\Engineerings;

return new class extends Migration {
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
                'name'         => 'add_eng',
                'display_name' => 'Add Engineerings',
            ],
            [
                'name'         => 'view_eng',
                'display_name' => 'View Engineerings',
            ],
            [
                'name'         => 'edit_eng',
                'display_name' => 'Edit Engineerings',
            ],
            [
                'name'         => 'delete_eng',
                'display_name' => 'Delete Engineerings',
            ],
        ];

        $module = Module::where('module_name', 'engineerings')->first();
        if (!$module) {
            $module = new Module();
            $module->module_name = 'engineerings';
            $module->description = 'User can view all Engineerings.';
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

        $all = ['add_eng', 'view_eng', 'edit_eng', 'delete_eng'];
        Permission::whereIn('name', $all)->update(['allowed_permissions' => Permission::ALL_NONE]);

        $companies = Company::all();

        foreach ($companies as $company) {
            $roles = ['employee', 'admin'];
            ModuleSetting::createRoleSettingEntry('engineerings', $roles, $company);
        }

        Artisan::call('module:enable engineerings');
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        $module = Module::where('module_name', 'engineerings')->first();
        $permisisons = Permission::where('module_id', $module->id)->get()->pluck('id')->toArray();
        PermissionRole::whereIn('permission_id', $permisisons)->delete();

        Module::where('module_name', 'engineerings')->delete();
        ModuleSetting::where('module_name', 'engineerings')->delete();
    }
};
