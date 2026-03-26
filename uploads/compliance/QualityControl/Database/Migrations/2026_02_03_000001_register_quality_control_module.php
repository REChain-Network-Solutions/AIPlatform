<?php

use Illuminate\Database\Migrations\Migration;
use App\Models\Module;
use App\Models\Permission;

return new class extends Migration {
    public function up(): void
    {
        $permissions = [
            ['name' => 'view_quality_control', 'display_name' => 'View Quality Control'],
            ['name' => 'add_quality_control', 'display_name' => 'Add Quality Control'],
            ['name' => 'edit_quality_control', 'display_name' => 'Edit Quality Control'],
            ['name' => 'delete_quality_control', 'display_name' => 'Delete Quality Control'],
        ];

        $module = Module::firstOrCreate(
            ['module_name' => 'quality_control'],
            ['is_superadmin' => 0, 'description' => 'Quality Control module']
        );

        // Create permissions idempotently
        foreach ($permissions as $perm) {
            $module->permissions()->firstOrCreate(
                ['name' => $perm['name']],
                $perm
            );
        }

        // Do not backfill module_settings here; use activation command.
    }

    public function down(): void
    {
        // Non-destructive down (do not delete modules/permissions in Worksuite deployments)
    }
};
