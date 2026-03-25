<?php

namespace Modules\PMCore\Database\Seeders;

use Illuminate\Database\Seeder;
use Modules\PMCore\app\Models\ProjectStatus;

class ProjectStatusSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $statuses = [
            [
                'name' => 'Planning',
                'slug' => 'planning',
                'color' => '#6f42c1',
                'position' => 1,
                'is_default' => true,
            ],
            [
                'name' => 'In Progress',
                'slug' => 'in_progress',
                'color' => '#007bff',
                'position' => 2,
                'is_default' => false,
            ],
            [
                'name' => 'On Hold',
                'slug' => 'on_hold',
                'color' => '#ffc107',
                'position' => 3,
                'is_default' => false,
            ],
            [
                'name' => 'Completed',
                'slug' => 'completed',
                'color' => '#28a745',
                'position' => 4,
                'is_default' => false,
            ],
            [
                'name' => 'Cancelled',
                'slug' => 'cancelled',
                'color' => '#dc3545',
                'position' => 5,
                'is_default' => false,
            ],
        ];

        foreach ($statuses as $status) {
            ProjectStatus::firstOrCreate(
                ['slug' => $status['slug']],
                $status
            );
        }
    }
}
