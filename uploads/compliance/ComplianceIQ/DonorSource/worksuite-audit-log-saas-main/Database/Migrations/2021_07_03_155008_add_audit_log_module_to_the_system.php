<?php

use App\Module;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class AddAuditLogModuleToTheSystem extends Migration
{

    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        $module = Module::create(['module_name' => 'auditlog', 'description' => 'Audit Log is a module to track activity for all users']);
        $module->permissions()->create(['name' => 'view_auditlog', 'display_name' => 'View Audit Log Data']);;
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Module::where('module_name', 'auditlog')->first()->delete();
    }
}
