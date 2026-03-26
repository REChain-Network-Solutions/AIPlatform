<?php

namespace Modules\QualityControl\Console\Commands;

use Illuminate\Console\Command;
use App\Models\Company;
use App\Models\Module;
use App\Models\Permission;
use Modules\QualityControl\Entities\RecurringSchedule;

class ActivateQualityControlModuleCommand extends Command
{
    protected $signature = 'quality-control:activate';
    protected $description = 'Backfill Quality Control module settings for all companies';

    public function handle(): int
    {
        $module = Module::firstOrCreate(
            ['module_name' => 'quality_control'],
            ['is_superadmin' => 0, 'description' => 'Quality Control module']
        );

        $permissions = [
            ['name' => 'view_quality_control', 'display_name' => 'View Quality Control'],
            ['name' => 'add_quality_control', 'display_name' => 'Add Quality Control'],
            ['name' => 'edit_quality_control', 'display_name' => 'Edit Quality Control'],
            ['name' => 'delete_quality_control', 'display_name' => 'Delete Quality Control'],
        ];

        foreach ($permissions as $perm) {
            // Relationship exists in Worksuite's Module model; fallback to direct table if not.
            try {
                $module->permissions()->firstOrCreate(['name' => $perm['name']], $perm);
            } catch (\Throwable $e) {
                Permission::firstOrCreate(
                    ['name' => $perm['name']],
                    $perm
                );
            }
        }

        $companies = Company::all();
        foreach ($companies as $company) {
            RecurringSchedule::addModuleSetting($company);
        }

        $this->info('Quality Control backfill complete for ' . $companies->count() . ' companies.');
        return self::SUCCESS;
    }
}
