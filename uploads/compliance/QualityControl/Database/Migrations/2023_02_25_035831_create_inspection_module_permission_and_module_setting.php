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
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;
use Modules\QualityControl\Entities\RecurringSchedule;

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
                'name' => 'add_quality_control',
                'display_name' => 'Add Quality Control'
            ],
            [
                'name' => 'view_quality_control',
                'display_name' => 'View Quality Control'
            ],
            [
                'name' => 'edit_quality_control',
                'display_name' => 'Edit Quality Control'
            ],
            [
                'name' => 'delete_quality_control',
                'display_name' => 'Delete Quality Control'
            ]
        ];

        $module = Module::firstOrCreate(
            ['module_name' => 'quality_control'],
            ['description' => 'Quality Control module']
        );

        // Create permissions idempotently
        foreach ($permissions as $perm) {
            $module->permissions()->firstOrCreate(['name' => $perm['name']], $perm);
        }

        $all = ['add_quality_control', 'view_quality_control', 'edit_quality_control', 'delete_quality_control'];
        Permission::whereIn('name', $all)->update(['allowed_permissions' => Permission::ALL_NONE]);

        // module_settings backfill is handled by activation command


    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        $module = Module::where('module_name', 'quality_control')->first();
        if (!$module) {
            return;
        }

        $permissionIds = Permission::where('module_id', $module->id)->pluck('id')->toArray();
        PermissionRole::whereIn('permission_id', $permissionIds)->delete();
        ModuleSetting::where('module_name', 'quality_control')->delete();
        Module::where('module_name', 'quality_control')->delete();
    }
};