<?php

namespace Database\Seeders;

use App\Models\Crm\CrmPipeline;
use App\Models\Crm\CrmPipelineStage;
use App\Models\User;
use Illuminate\Database\Seeder;

/**
 * Creates a default "Sales Pipeline" for the first admin user.
 * Run: php artisan db:seed --class=BosDefaultPipelineSeeder
 */
class BosDefaultPipelineSeeder extends Seeder
{
    public function run(): void
    {
        $admin = User::query()->where('type', 'admin')->first();

        if (! $admin) {
            $this->command->warn('No admin user found. Skipping default pipeline creation.');

            return;
        }

        $pipeline = CrmPipeline::firstOrCreate(
            ['user_id' => $admin->id, 'name' => 'Sales Pipeline'],
            ['currency' => 'USD', 'is_default' => true, 'order' => 0]
        );

        $stages = [
            ['name' => 'Lead',       'color' => '#94a3b8', 'probability' => 10, 'order' => 0],
            ['name' => 'Qualified',  'color' => '#3b82f6', 'probability' => 25, 'order' => 1],
            ['name' => 'Proposal',   'color' => '#f59e0b', 'probability' => 50, 'order' => 2],
            ['name' => 'Negotiation','color' => '#f97316', 'probability' => 75, 'order' => 3],
            ['name' => 'Won',        'color' => '#22c55e', 'probability' => 100, 'order' => 4, 'is_won' => true],
            ['name' => 'Lost',       'color' => '#ef4444', 'probability' => 0,   'order' => 5, 'is_lost' => true],
        ];

        foreach ($stages as $stage) {
            CrmPipelineStage::firstOrCreate(
                ['pipeline_id' => $pipeline->id, 'name' => $stage['name']],
                array_merge($stage, ['pipeline_id' => $pipeline->id])
            );
        }

        $this->command->info("Default pipeline created for: {$admin->email}");
    }
}
