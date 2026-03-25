<?php

namespace Modules\PMCore\database\seeders;

use App\Models\ModuleSetting;
use Illuminate\Database\Seeder;

class PMCoreSettingsSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $settings = [
            // Project Defaults
            'default_project_status' => 'planning',
            'default_project_priority' => 'medium',
            'default_is_billable' => true,

            // Project Code Generation
            'auto_generate_codes' => true,
            'code_prefix_length' => '3',
            'code_separator' => '-',

            // Timesheet Rules
            'allow_future_timesheets' => false,
            'max_hours_per_day' => '12',
            'require_task_assignment' => false,

            // Budget Management
            'budget_warning_threshold' => '80',
            'budget_alert_threshold' => '100',
            'default_currency_symbol' => '$',

            // Team Management
            'default_allocation_percentage' => '100',
            'allow_multiple_project_managers' => false,
        ];

        foreach ($settings as $key => $value) {
            ModuleSetting::firstOrCreate(
                [
                    'module' => 'PMCore',
                    'key' => $key,
                ],
                [
                    'value' => is_bool($value) ? ($value ? '1' : '0') : (string) $value,
                    'created_at' => now(),
                    'updated_at' => now(),
                ]
            );
        }
    }
}
