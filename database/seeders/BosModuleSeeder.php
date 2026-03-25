<?php

namespace Database\Seeders;

use App\Models\Bos\BosModule;
use Illuminate\Database\Seeder;

/**
 * Seeds the core Titan BOS module registry.
 * Run: php artisan db:seed --class=BosModuleSeeder
 */
class BosModuleSeeder extends Seeder
{
    public function run(): void
    {
        $modules = [
            [
                'key'          => 'crm',
                'name'         => 'CRM',
                'description'  => 'Manage contacts, companies, deals, and your sales pipeline.',
                'icon'         => 'tabler-users-group',
                'color'        => '#6366f1',
                'route_prefix' => 'crm',
                'is_active'    => true,
                'is_core'      => true,
                'order'        => 1,
            ],
            [
                'key'          => 'hr',
                'name'         => 'HR',
                'description'  => 'Employee directory, time tracking, and leave management.',
                'icon'         => 'tabler-id-badge',
                'color'        => '#f59e0b',
                'route_prefix' => 'hr',
                'is_active'    => false, // Phase 3
                'is_core'      => false,
                'order'        => 2,
            ],
            [
                'key'          => 'projects',
                'name'         => 'Projects',
                'description'  => 'Task management, milestones, and team collaboration.',
                'icon'         => 'tabler-layout-kanban',
                'color'        => '#10b981',
                'route_prefix' => 'projects',
                'is_active'    => false, // Phase 3
                'is_core'      => false,
                'order'        => 3,
            ],
            [
                'key'          => 'finance',
                'name'         => 'Finance',
                'description'  => 'Invoicing, expenses, P&L, and financial reporting.',
                'icon'         => 'tabler-chart-pie',
                'color'        => '#3b82f6',
                'route_prefix' => 'finance',
                'is_active'    => false, // Phase 2
                'is_core'      => false,
                'order'        => 4,
            ],
            [
                'key'          => 'intelligence',
                'name'         => 'AI Intelligence',
                'description'  => 'AI-powered reports, forecasting, and business insights.',
                'icon'         => 'tabler-brain',
                'color'        => '#8b5cf6',
                'route_prefix' => 'intelligence',
                'is_active'    => false, // Phase 4
                'is_core'      => false,
                'order'        => 5,
            ],
        ];

        foreach ($modules as $module) {
            BosModule::updateOrCreate(['key' => $module['key']], $module);
        }
    }
}
