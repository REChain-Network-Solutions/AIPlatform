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
use Modules\Inspection\Entities\RecurringSchedule;

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
                'name' => 'add_inspection',
                'display_name' => 'Add Inspection'
            ],
            [
                'name' => 'view_inspection',
                'display_name' => 'View Inspection'
            ],
            [
                'name' => 'edit_inspection',
                'display_name' => 'Edit Inspection'
            ],
            [
                'name' => 'delete_inspection',
                'display_name' => 'Delete Inspection'
            ]
        ];

        $module = new Module();
        $module->module_name = 'inspection';
        $module->description = 'User can view all.';
        $module->saveQuietly();

        $module->permissions()->createMany($permissions);

        $all = ['add_inspection', 'view_inspection', 'edit_inspection', 'delete_inspection'];
        Permission::whereIn('name', $all)->update(['allowed_permissions' => Permission::ALL_NONE]);

        $companies = Company::all();

        // We will insert these for the new company from event listener
        foreach ($companies as $company) {
            RecurringSchedule::addModuleSetting($company);
        }

        Artisan::call('module:enable inspection');

    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {

        $module = Module::where('module_name', 'inspection')->first();
        $permisisons = Permission::where('module_id', $module->id)->get()->pluck('id')->toArray();
        PermissionRole::whereIn('permission_id', $permisisons)->delete();
        ModuleSetting::where('module_name', 'inspection')->delete();
        Module::where('module_name', 'inspection')->delete();
    }
};
